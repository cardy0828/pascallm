// lfm3.cu
//
// LFM3 の 1 トークン生成（decode）に使うカーネルです（M2）。Go 側（internal/lfm3gpu）から名前で呼ぶため、
// すべて extern "C" で公開します。ビルドは kernels/build.sh（sm_61 の cubin）で行います。
//
// 共通の約束:
//   - 「状態」state は int32 の 4 要素です。[0] 入力トークン、[1] 位置（0 始まり）、[2] 直前の argmax の結果。
//     カーネルは位置とトークンを state から読むため、1 トークン分の処理を CUDA Graph に 1 回記録すれば、
//     位置が変わっても同じ Graph を再生できます。
//   - 活性値の int8 化は 32 要素ごとで、1 ワープ（32 スレッド）が 1 ブロックを担当します。
//     手順は CPU 版（internal/pq4 の QuantizeActivations）と同じです。
//   - PQ4 の重みは、4bit 本体 [行][ブロック][16 バイト] と FP16 係数 [行][ブロック] の分離配置です。

#include <cstdint>
#include <cuda_fp16.h>

namespace {

constexpr int kWarpSize = 32;
constexpr unsigned kFullMask = 0xffffffffu;
constexpr int kQuantBlock = 32;          // 活性値・重みのブロックの要素数
constexpr float kActivationLevels = 127.0f;  // int8 化で使う最大の整数値
constexpr float kWeightCenter = 8.0f;    // 4bit 重みの中心（復元値は (q - 8) × d）
constexpr int kGemvWarpsPerBlock = 4;    // GEMV の 1 スレッドブロックのワープ数
constexpr int kGemvRowsPerWarp = 4;      // GEMV の 1 ワープが担当する行数
constexpr int kNormMaxPerThread = 4;     // residual_norm_quantize で 1 スレッドが持つ最大の要素数
constexpr int kConvMaxKernel = 8;        // short_conv_quantize が対応する最大の畳み込みの長さ（LFM3 は 7）

// attention_decode の寸法です。
constexpr int kAttentionHeadDim = 128;   // 対応する head_dim
constexpr int kAttentionGroup = 4;       // 1 個の KV ヘッドを共有する query ヘッドの数（num_heads / num_kv_heads）
constexpr int kAttentionThreads = 256;   // 1 スレッドブロックのスレッド数
constexpr int kAttentionWarps = kAttentionThreads / kWarpSize;            // 8
constexpr int kAttentionChunk = 64;      // 1 回にまとめて読む位置数（これより長い範囲は区切って処理）
constexpr int kLanesPerPosition = 16;    // score・重み付き和で 1 位置を受け持つレーン数
constexpr int kDimsPerScoreLane = kAttentionHeadDim / kLanesPerPosition;  // 1 レーンが受け持つ次元数（8）
constexpr int kPositionsPerPass = kAttentionThreads / kLanesPerPosition;  // 同時に処理する位置数（16）
constexpr int kPassesPerChunk = kAttentionChunk / kPositionsPerPass;      // 1 区切りを処理する回数（4）
constexpr int kAttentionMinPositionsPerSplit = 128;  // 位置の範囲を分割するとき、1 分割に入れる最小の位置数
constexpr int kAttentionPartialStride = 2 + kAttentionHeadDim;  // 部分結果 1 個の大きさ: 最大値・合計・重み付き和 128
constexpr int kPrologueThreadsPerHead = kAttentionThreads / kAttentionGroup;  // 前処理で 1 ヘッドを受け持つスレッド数（64）
constexpr int kPrologueWarpsPerHead = kPrologueThreadsPerHead / kWarpSize;     // 2
constexpr int kPrologueDims = kAttentionHeadDim / kPrologueThreadsPerHead;     // 前処理で 1 スレッドが受け持つ次元数（2）

// 入力処理（prefill、M4）の寸法です。
constexpr int kGemmTileRows = 64;      // gemm_pq4 の 1 ブロックが受け持つ重みの行数
constexpr int kGemmTileTokens = 128;   // gemm_pq4 の 1 ブロックが受け持つトークン数
constexpr int kGemmThreads = 256;      // gemm_pq4 のスレッド数
constexpr int kGemmThreadRows = 4;     // 1 スレッドが受け持つ行数（行 rx, rx+16, rx+32, rx+48）
constexpr int kGemmThreadTokens = 8;   // 1 スレッドが受け持つトークン数（トークン ty, ty+16, …, ty+112）
constexpr int kGemmRowGroups = kGemmTileRows / kGemmThreadRows;        // 16
constexpr int kGemmStageBlocks = 4;    // 共有メモリに 1 回で置く K 方向のブロック数（128 要素）
constexpr int kPrefillTokensPerBlock = 8;  // attention_prefill の 1 ブロックが受け持つトークン数（1 ワープ 1 トークン）
constexpr int kPrefillChunk = 32;          // attention_prefill が 1 回に共有メモリに読む位置数（1 レーン 1 位置）
constexpr int kPrefillKeyStride = kAttentionHeadDim + 2;  // 共有メモリの K の 1 行の FP16 数（バンクの衝突を避ける詰め物 2 個）

// 量子化済み活性値 1 ブロックの付帯情報です（CPU 版の pq4.ActivationBlock と同じ）。
// scale: x ≒ scale × q、scaled_sum: scale × Σq（重みの中心 8 を引く補正に使う）
struct ActBlockInfo {
    float scale;
    float scaled_sum;
};

// warp_sum はワープ内 32 レーンの値の合計を、全レーンに返します。
__device__ __forceinline__ float warp_sum(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(kFullMask, value, offset);
    }
    return value;
}

// warp_sum_int は整数版の warp_sum です。
__device__ __forceinline__ int warp_sum_int(int value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(kFullMask, value, offset);
    }
    return value;
}

// warp_max はワープ内 32 レーンの最大値を、全レーンに返します。
__device__ __forceinline__ float warp_max(float value) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(kFullMask, value, offset));
    }
    return value;
}

// block_sum はスレッドブロック全体の値の合計を、全スレッドに返します。
// blockDim.x は 32 の倍数（最大 1024）であること。shared は 32 要素以上の共有メモリです。
// 例: 256 スレッドがそれぞれ 1 を渡すと、全員が 256 を受け取る。
__device__ float block_sum(float value, float* shared) {
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int warps = blockDim.x / kWarpSize;
    value = warp_sum(value);
    __syncthreads();  // 直前の block_sum が shared を読み終わるのを待つ
    if (lane == 0) shared[warp] = value;
    __syncthreads();
    float total = lane < warps ? shared[lane] : 0.0f;
    return warp_sum(total);
}

// quantize_warp は、ワープの各レーンが持つ 1 要素（計 32 要素）を int8 に量子化し、
// q[block_index × 32 + lane] と info[block_index] に書きます。ワープの全レーンで呼ぶこと。
// 例: 最大絶対値が 2.54 なら scale = 0.02、値 1.0 は round(50) = 50
__device__ void quantize_warp(float value, int block_index, int8_t* __restrict__ q,
                              ActBlockInfo* __restrict__ info) {
    const int lane = threadIdx.x % kWarpSize;
    const float max_abs = warp_max(fabsf(value));
    const float scale = max_abs / kActivationLevels;
    const float inverse = scale > 0.0f ? 1.0f / scale : 0.0f;
    const int quantized = __float2int_rn(value * inverse);  // 偶数丸め（CPU 版の RoundToEven と同じ）
    q[block_index * kQuantBlock + lane] = static_cast<int8_t>(quantized);
    const int sum = warp_sum_int(quantized);
    if (lane == 0) info[block_index] = {scale, scale * static_cast<float>(sum)};
}

// low_nibbles / high_nibbles は、4 バイトの各バイトの下位・上位 4bit を取り出します。
// 例: 0x21436587 -> 下位 0x01030507、上位 0x02040608
__device__ __forceinline__ int low_nibbles(int packed) { return packed & 0x0F0F0F0F; }
__device__ __forceinline__ int high_nibbles(int packed) { return (packed >> 4) & 0x0F0F0F0F; }

// sigmoid は 1 / (1 + exp(-x)) です。例: sigmoid(0) = 0.5
__device__ __forceinline__ float sigmoid(float x) { return 1.0f / (1.0f + expf(-x)); }

// rotate_half_at は、RoPE（rotate_half 方式）を掛けた後の、次元 d の値を返します。vector は回転前の head_dim 個の値、
// cosine・sine は次元 d の角度のものです。前半の次元 d は後半の d + head_dim/2 と組になります。
// 例: head_dim = 128、d = 3 なら vector[3]·cos − vector[67]·sin、d = 67 なら vector[67]·cos + vector[3]·sin
__device__ __forceinline__ float rotate_half_at(const float* vector, int d, float cosine, float sine) {
    const int half_dim = kAttentionHeadDim / 2;
    return d < half_dim ? vector[d] * cosine - vector[d + half_dim] * sine
                        : vector[d] * cosine + vector[d - half_dim] * sine;
}

// unpack_halves は、uint4（FP16 × 8）を float 8 個にして out[0..7] に書きます。
// 例: FP16 の 1.0, 2.0, ... が入っていれば out = {1, 2, ...}
__device__ __forceinline__ void unpack_halves(uint4 packed, float* out) {
    const __half2* pairs = reinterpret_cast<const __half2*>(&packed);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 values = __half22float2(pairs[j]);
        out[2 * j] = values.x;
        out[2 * j + 1] = values.y;
    }
}

// 選び方の候補を GPU で絞る処理（candidates_partial・candidates_final）の寸法です。
// 候補の並べ替えは、1 ワープが kCandidateCount 個（1 レーン kCandidatesPerLane 個）をレジスタに持って行い、
// ワープ間は共有メモリを介して「2 つの上位 128 個から上位 128 個を作る」ことを木の形で繰り返します。
constexpr int kCandidateCount = 128;                                     // 最後に CPU に渡す候補の数
constexpr int kCandidatesPerLane = kCandidateCount / kWarpSize;          // 1 レーンが持つ候補の数（4）
constexpr int kCandidatePartThreads = 512;                               // 1 段目の 1 ブロックのスレッド数
constexpr int kCandidatePartWarps = kCandidatePartThreads / kWarpSize;   // 1 段目の 1 ブロックのワープ数（16）
constexpr int kCandidateFinalThreads = 1024;                             // 2 段目のスレッド数
constexpr int kCandidateFinalWarps = kCandidateFinalThreads / kWarpSize;  // 2 段目のワープ数（32）
constexpr int kCandidateMissingId = INT32_MAX;                           // 区間の外を表す番号（並べ替えで最後になる）
// 1 段目のブロック数は 2 × kCandidateFinalWarps（64）で、2 段目の各ワープが 2 区間ずつ受け持ちます。
// 1 区間の長さの上限は kCandidatePartWarps × kCandidateCount（2048）です（Go 側 candidateParts・candidatePartSpan）。

// Candidate は候補 1 個（ペナルティを適用した値と番号）です。
struct Candidate {
    float value;
    int id;
};

// candidate_before は、候補 a を b より前に置くか（値が大きい、同じ値なら番号が小さい）を返します。
// 例: (2.0, 7) と (2.0, 3) → (2.0, 3) が前
__device__ __forceinline__ bool candidate_before(Candidate a, Candidate b) {
    return a.value > b.value || (a.value == b.value && a.id < b.id);
}

// better_candidate は、a と b のうち前に置く方を返します。
__device__ __forceinline__ Candidate better_candidate(Candidate a, Candidate b) {
    return candidate_before(b, a) ? b : a;
}

// warp_bitonic_step は、ワープの kCandidateCount 個に bitonic の 1 段（区間 size・間隔 stride）を適用します。
// 要素の位置は「レーン × kCandidatesPerLane + 添字」です。stride が kCandidatesPerLane 以上ならレーン間で
// shuffle で交換し、未満ならレーン内のレジスタで交換します。size の区間ごとに最良を先にする区間と最後にする区間を
// 交互に作ります（size = kCandidateCount なら全体を最良が先の順にする）。ワープの全レーンで呼びます。
__device__ __forceinline__ void warp_bitonic_step(Candidate (&items)[kCandidatesPerLane], int size, int stride) {
    const int lane = threadIdx.x % kWarpSize;
    if (stride >= kCandidatesPerLane) {
        const int lane_stride = stride / kCandidatesPerLane;
        const bool lower = (lane & lane_stride) == 0;  // 組の小さい位置の側か
#pragma unroll
        for (int j = 0; j < kCandidatesPerLane; ++j) {
            const int position = lane * kCandidatesPerLane + j;
            const bool best_first = (position & size) == 0;
            const Candidate other = {__shfl_xor_sync(kFullMask, items[j].value, lane_stride),
                                     __shfl_xor_sync(kFullMask, items[j].id, lane_stride)};
            // 小さい位置の側は、best_first なら良い方を、そうでなければ悪い方を持ちます（大きい位置の側は逆）。
            const bool keep_better = lower == best_first;
            if (candidate_before(other, items[j]) == keep_better) {
                items[j] = other;
            }
        }
        return;
    }
#pragma unroll
    for (int j = 0; j < kCandidatesPerLane; ++j) {
        const int partner = j ^ stride;
        if (partner > j) {
            const int position = lane * kCandidatesPerLane + j;
            const bool best_first = (position & size) == 0;
            if (candidate_before(items[partner], items[j]) == best_first) {
                const Candidate swapped = items[j];
                items[j] = items[partner];
                items[partner] = swapped;
            }
        }
    }
}

// warp_sort_candidates は、ワープの kCandidateCount 個を最良が先の順に並べます（bitonic sort）。
__device__ __forceinline__ void warp_sort_candidates(Candidate (&items)[kCandidatesPerLane]) {
#pragma unroll
    for (int size = 2; size <= kCandidateCount; size *= 2) {
#pragma unroll
        for (int stride = size / 2; stride > 0; stride /= 2) {
            warp_bitonic_step(items, size, stride);
        }
    }
}

// warp_merge_candidates は、items（最良が先の順の上位 kCandidateCount 個）と、other（同じ順の別の列）から、
// 合わせた中の上位 kCandidateCount 個を最良が先の順で items に作ります。
// 位置 p で items[p] と other[127 - p] の良い方を取ると、上位 128 個からなる bitonic な列になり、
// それを bitonic merge（間隔 64, 32, …, 1）で並べます。
__device__ __forceinline__ void warp_merge_candidates(Candidate (&items)[kCandidatesPerLane],
                                                      const Candidate* __restrict__ other) {
    const int lane = threadIdx.x % kWarpSize;
#pragma unroll
    for (int j = 0; j < kCandidatesPerLane; ++j) {
        const int position = lane * kCandidatesPerLane + j;
        items[j] = better_candidate(items[j], other[kCandidateCount - 1 - position]);
    }
#pragma unroll
    for (int stride = kCandidateCount / 2; stride > 0; stride /= 2) {
        warp_bitonic_step(items, kCandidateCount, stride);
    }
}

// merge_warp_candidates は、ブロックの warps 個のワープが持つ上位の列を 1 本にまとめ、ワープ 0 の items に残します。
// 後ろ半分のワープが列を共有メモリ lists（warps / 2 × kCandidateCount 個）に置き、前半分のワープが自分の列と
// まとめる、を 1 本になるまで繰り返します。warps は 2 のべき乗です。ブロックの全スレッドで呼びます。
__device__ void merge_warp_candidates(Candidate (&items)[kCandidatesPerLane], Candidate* lists, int warps) {
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    for (int active = warps; active > 1; active /= 2) {
        const int half = active / 2;
        if (warp >= half && warp < active) {
#pragma unroll
            for (int j = 0; j < kCandidatesPerLane; ++j) {
                lists[(warp - half) * kCandidateCount + lane * kCandidatesPerLane + j] = items[j];
            }
        }
        __syncthreads();
        if (warp < half) {
            warp_merge_candidates(items, lists + warp * kCandidateCount);
        }
        __syncthreads();  // 次の回で lists を書き換える前に、読み終えるのを待つ
    }
}

// block_reduce_max は、ブロックの全スレッドの value の最大値を、全スレッドに返します。
// scratch は共有メモリの float × 32 です。ブロックの全スレッドで呼びます。
__device__ float block_reduce_max(float value, float* scratch) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(kFullMask, value, offset));
    }
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    value = lane < blockDim.x / kWarpSize ? scratch[lane] : -INFINITY;
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value = fmaxf(value, __shfl_xor_sync(kFullMask, value, offset));
    }
    __syncthreads();  // scratch を次に使う前に、全ワープが読み終えるのを待つ
    return value;
}

// block_reduce_sum は、ブロックの全スレッドの value の合計を、全スレッドに返します（block_reduce_max と同じ使い方）。
__device__ float block_reduce_sum(float value, float* scratch) {
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(kFullMask, value, offset);
    }
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    if (lane == 0) {
        scratch[warp] = value;
    }
    __syncthreads();
    value = lane < blockDim.x / kWarpSize ? scratch[lane] : 0.0f;
    for (int offset = kWarpSize / 2; offset > 0; offset /= 2) {
        value += __shfl_xor_sync(kFullMask, value, offset);
    }
    __syncthreads();
    return value;
}

// penalized_logit は、CPU 版（internal/sampling の applyPenalties）と同じ順でペナルティを適用します。
//   - seen（プロンプトか生成済み）なら repetition_penalty: 正なら割り、0 以下なら掛ける
//   - count（生成した回数）が 1 以上なら presence_penalty + frequency_penalty × count を引く
// 例: logit 4、seen、repetition 2、count 0 → 2
__device__ __forceinline__ float penalized_logit(float logit, bool seen, int count, float repetition_penalty,
                                                 float presence_penalty, float frequency_penalty) {
    if (seen) {
        logit = logit > 0.0f ? logit / repetition_penalty : logit * repetition_penalty;
    }
    if (count > 0) {
        logit -= presence_penalty + frequency_penalty * static_cast<float>(count);
    }
    return logit;
}

}  // namespace

extern "C" {

// embed_lookup は、state[0] のトークンの埋め込み（FP16）を FP32 にして hidden に書きます。
// 起動: grid = hidden / 256、block = 256。例: hidden = 2048 なら grid = 8
__global__ void embed_lookup(const __half* __restrict__ table, const int* __restrict__ state,
                             float* __restrict__ hidden, int hidden_size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_size) return;
    const int token = state[0];
    hidden[index] = __half2float(table[static_cast<size_t>(token) * hidden_size + index]);
}

// residual_norm_quantize は、残差の加算と次の正規化・int8 化をまとめて行います。
//
//   delta が null でなければ: hidden += RMSNorm(delta) × delta_weight（sandwich norm の後段の正規化）
//   normed = RMSNorm(hidden) × next_weight を FP32 で書き、32 要素ごとに int8 化して q・info に書く
//
// 起動: grid = 1、block = 1024（1 スレッドブロックでベクトル全体を処理）。
// n は 32 の倍数で、kNormMaxPerThread × 1024 以下。例: n = 2048 なら 1 スレッド 2 要素
//
// 速度のための工夫: 1 スレッドブロックしか使わないため、グローバルメモリの読み出しを 1 要素ずつ待つと遅くなります
// （最初の版は 1 回 約 29 µs）。そこで、必要な値をすべて最初にまとめて読み、レジスタに置いたまま計算します。
// スレッド t は要素 t, t+1024, ... を受け持つので、1 ワープの 32 レーンは連続した 32 要素（int8 化の 1 ブロック）を
// 持ちます。そのため int8 化も、書き戻した値を読み直さずにレジスタから行えます。
// 本体は residual_norm_quantize_vector（__device__ 関数）で、生成用のカーネル residual_norm_quantize と、
// 入力処理用の residual_norm_quantize_rows の両方から呼び、同じ数値にします。
__device__ void residual_norm_quantize_vector(float* __restrict__ hidden, const float* __restrict__ delta,
                                              const float* __restrict__ delta_weight,
                                              const float* __restrict__ next_weight, float eps, int n,
                                              float* __restrict__ normed, int8_t* __restrict__ q,
                                              ActBlockInfo* __restrict__ info) {
    __shared__ float shared[kWarpSize];
    const bool has_delta = delta != nullptr;
    float residual[kNormMaxPerThread];
    float update[kNormMaxPerThread];
    float update_weight[kNormMaxPerThread];
    float output_weight[kNormMaxPerThread];
#pragma unroll
    for (int k = 0; k < kNormMaxPerThread; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        const bool valid = i < n;
        residual[k] = valid ? hidden[i] : 0.0f;
        update[k] = valid && has_delta ? delta[i] : 0.0f;
        update_weight[k] = valid && has_delta ? delta_weight[i] : 0.0f;
        output_weight[k] = valid ? next_weight[i] : 0.0f;
    }
    if (has_delta) {
        float squares = 0.0f;
#pragma unroll
        for (int k = 0; k < kNormMaxPerThread; ++k) squares += update[k] * update[k];
        const float inverse_rms = 1.0f / sqrtf(block_sum(squares, shared) / n + eps);
#pragma unroll
        for (int k = 0; k < kNormMaxPerThread; ++k) {
            const int i = threadIdx.x + k * blockDim.x;
            if (i < n) {
                residual[k] += update_weight[k] * (update[k] * inverse_rms);
                hidden[i] = residual[k];
            }
        }
    }
    float squares = 0.0f;
#pragma unroll
    for (int k = 0; k < kNormMaxPerThread; ++k) squares += residual[k] * residual[k];
    const float inverse_rms = 1.0f / sqrtf(block_sum(squares, shared) / n + eps);
#pragma unroll
    for (int k = 0; k < kNormMaxPerThread; ++k) {
        const int i = threadIdx.x + k * blockDim.x;
        // n と blockDim.x は 32 の倍数なので、i < n はワープの全レーンで同じ結果になる（quantize_warp の条件）。
        if (i < n) {
            const float value = output_weight[k] * (residual[k] * inverse_rms);
            normed[i] = value;
            quantize_warp(value, i / kQuantBlock, q, info);
        }
    }
}

// residual_norm_quantize は、生成（1 トークン）用の残差・正規化・int8 化です（本体は上の関数）。
// 起動: grid = 1、block = 1024
__global__ void residual_norm_quantize(float* __restrict__ hidden, const float* __restrict__ delta,
                                       const float* __restrict__ delta_weight,
                                       const float* __restrict__ next_weight, float eps, int n,
                                       float* __restrict__ normed, int8_t* __restrict__ q,
                                       ActBlockInfo* __restrict__ info) {
    residual_norm_quantize_vector(hidden, delta, delta_weight, next_weight, eps, n, normed, q, info);
}

// pq4_block_dot は、PQ4 の重み 1 ブロック（32 要素、16 バイト）と int8 の活性値 1 ブロック（a0・a1 の 32 バイト）の
// 整数の内積 Σ q_w·q_a を返します（重みの中心 8 は引かない。呼び出し側で scaled_sum を使って補正する）。
// バイト j の下位 4bit が要素 j（a0 側）、上位 4bit が要素 j+16（a1 側）です。
__device__ __forceinline__ int pq4_block_dot(int4 w, int4 a0, int4 a1) {
    int dot = 0;
    dot = __dp4a(low_nibbles(w.x), a0.x, dot);
    dot = __dp4a(low_nibbles(w.y), a0.y, dot);
    dot = __dp4a(low_nibbles(w.z), a0.z, dot);
    dot = __dp4a(low_nibbles(w.w), a0.w, dot);
    dot = __dp4a(high_nibbles(w.x), a1.x, dot);
    dot = __dp4a(high_nibbles(w.y), a1.y, dot);
    dot = __dp4a(high_nibbles(w.z), a1.z, dot);
    dot = __dp4a(high_nibbles(w.w), a1.w, dot);
    return dot;
}

// gemv_pq4 は out = W·x を計算します。W は PQ4（分離配置）、x は int8 化した活性値です。
//
// 1 ワープが出力 kGemvRowsPerWarp（4）行を担当し、レーン l がブロック l, l+32, ... を読みます。ワープ内の 32 レーンが
// 1 行につき連続した 512 バイトを同時に読むため、メモリの読み出しがまとまります。4 行を同時に読むのは、
// 列が少ない行列（例: 2048 列 = 1 レーン 2 回）でも、待ちの間に出しておける読み出しを増やすためです
// （単独の計測で、2048×2048 が 8.7 → 8.2 µs、2048×7680 が 27.2 → 26.2 µs）。活性値の読み出しも 4 行で共有します。
//   Σ (q_w − 8)·d_w·d_a·q_a = d_w · (d_a·Σq_w·q_a − 8·d_a·Σq_a)
// PQ8（PQ4 のブロック 2 個で約 8bit、internal/pq4/pq8.go）も同じ式で計算できます。重みのブロック b が使う活性値の
// ブロックは b >> input_block_shift です（PQ4 は 0、PQ8 は 1 で、上位・下位の 2 ブロックが同じ活性値を使う）。
// blocks_per_row は 1 行あたりの重みのブロック数です。
// 起動: grid = ceil(rows / 16)、block = (32, 4)。rows は 4 の倍数。例: rows = 7680 なら grid = 480
__global__ void gemv_pq4(const int4* __restrict__ packed, const __half* __restrict__ scales,
                         const int* __restrict__ act_q, const ActBlockInfo* __restrict__ act_info,
                         float* __restrict__ out, int rows, int blocks_per_row, int input_block_shift) {
    const int first_row = (blockIdx.x * kGemvWarpsPerBlock + threadIdx.y) * kGemvRowsPerWarp;
    if (first_row >= rows) return;
    const int lane = threadIdx.x;
    const size_t first_offset = static_cast<size_t>(first_row) * blocks_per_row;
    float partial[kGemvRowsPerWarp];
#pragma unroll
    for (int r = 0; r < kGemvRowsPerWarp; ++r) partial[r] = 0.0f;
    for (int b = lane; b < blocks_per_row; b += kWarpSize) {
        int4 w[kGemvRowsPerWarp];
        float scale[kGemvRowsPerWarp];
#pragma unroll
        for (int r = 0; r < kGemvRowsPerWarp; ++r) {
            const size_t offset = first_offset + static_cast<size_t>(r) * blocks_per_row + b;
            w[r] = packed[offset];
            scale[r] = __half2float(scales[offset]);
        }
        const int input_block = b >> input_block_shift;
        const int4* act_vec = reinterpret_cast<const int4*>(act_q) + 2 * input_block;
        const int4 a0 = act_vec[0];
        const int4 a1 = act_vec[1];
        const ActBlockInfo info = act_info[input_block];
#pragma unroll
        for (int r = 0; r < kGemvRowsPerWarp; ++r) {
            const int dot = pq4_block_dot(w[r], a0, a1);
            partial[r] += scale[r] * (info.scale * static_cast<float>(dot) - kWeightCenter * info.scaled_sum);
        }
    }
#pragma unroll
    for (int r = 0; r < kGemvRowsPerWarp; ++r) partial[r] = warp_sum(partial[r]);
    if (lane == 0) {
#pragma unroll
        for (int r = 0; r < kGemvRowsPerWarp; ++r) out[first_row + r] = partial[r];
    }
}

// short_conv_quantize は、gated short convolution の畳み込み部分を計算し、結果を int8 化します。
//
//   [B, C, x] = conv_in（in_proj の出力、hidden ずつ並ぶ）、u = B × x
//   conv[c] = kernel[K-1][c]·u + Σ_{k<K-1} kernel[k][c]·（位置 p-(K-1)+k の u）
//   出力 = C × conv[c]
// 履歴は輪状バッファ history[K-1][hidden] で、位置 p の u は行 p mod (K-1) に入ります。位置 p-(K-1)+k の u は
// 行 (p+k) mod (K-1) にあり、位置 0 より前は 0（Reset で消す）です。読んだ後で、行 p mod (K-1)（位置 p-(K-1) の値。
// k = 0 で使い終わった）に現在の u を書きます。値をずらさないので、書き込みは 1 チャネル 1 回です。
// kernel は [K][hidden] に転置済みです。同じ行のチャネルが隣り合うので、ワープの読み出しがまとまります。
// 起動: grid = hidden / 256、block = 256（1 スレッド 1 チャネル、1 ワープが 32 チャネルを int8 化）
// 例: K = 7、位置 p = 10 なら、k = 0〜5 は行 4, 5, 0, 1, 2, 3（位置 4〜9）を読み、現在の u を行 4 に書く
__global__ void short_conv_quantize(const float* __restrict__ conv_in, float* __restrict__ history,
                                    const float* __restrict__ kernel, const int* __restrict__ state,
                                    int hidden_size, int kernel_size, int8_t* __restrict__ q,
                                    ActBlockInfo* __restrict__ info) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= hidden_size) return;  // hidden_size は 32 の倍数なので、ワープ単位で抜ける
    const int history_length = kernel_size - 1;
    const int position = state[1];
    // 係数と履歴は、直前の GEMV が重みを流した後でキャッシュに残っていないため、読み出しの待ちが長い。
    // 回数を定数（kConvMaxKernel - 1）にしてループを展開し、すべての読み出しを先に出して、待ちを 1 回分にまとめる。
    float past_taps[kConvMaxKernel - 1];
    float past[kConvMaxKernel - 1];
#pragma unroll
    for (int k = 0; k < kConvMaxKernel - 1; ++k) {
        const bool used = k < history_length;
        const int row = (position + k) % history_length;
        past_taps[k] = used ? kernel[static_cast<size_t>(k) * hidden_size + channel] : 0.0f;
        past[k] = used ? history[static_cast<size_t>(row) * hidden_size + channel] : 0.0f;
    }
    const float current_tap = kernel[static_cast<size_t>(history_length) * hidden_size + channel];
    const float gate_b = conv_in[channel];
    const float gate_c = conv_in[hidden_size + channel];
    const float input = conv_in[2 * hidden_size + channel];
    const float current = gate_b * input;
    // CPU 版と同じ順（現在の値、古い順の履歴）に足す。使わない k は係数・値とも 0 なので、足しても変わらない。
    float sum = current_tap * current;
#pragma unroll
    for (int k = 0; k < kConvMaxKernel - 1; ++k) sum += past_taps[k] * past[k];
    history[static_cast<size_t>(position % history_length) * hidden_size + channel] = current;
    quantize_warp(gate_c * sum, channel / kQuantBlock, q, info);
}

// attention_decode は、attention 層の q・k・v の GEMV の後から o_proj の前までを 1 カーネルで行います。
// 1 個の query（現在の位置）に対する因果 attention を計算し、ゲートを掛け、出力（num_heads × head_dim）を
// int8 化して q・info に書きます（o_proj の入力になる）。
//
// 前処理（各ブロックが自分の KV ヘッドの分を計算する。分割されたブロックは同じ計算を重ねて行う）:
//   - q = RMSNorm(qkv の q 部分) × q_norm に RoPE を掛ける（4 ヘッド分）
//   - k = RMSNorm(qkv の k 部分) × k_norm に RoPE を掛け、v はそのまま、FP16 にする（現在の位置の k・v）
//   - ヘッドごとのゲート sigmoid(gate_weight[h]·normed)（gate_weight は FP16）
//   - 現在の位置を受け持つブロックだけが、k・v を KV キャッシュの位置 state[1] に書く。
//     現在の位置の k・v は、キャッシュからではなく共有メモリから使う（他のブロックの書き込みを待たずに済む）
// RoPE の角度は「位置 × 逆周波数」（float32）です。逆周波数が 0 の次元は回転しません。
// 以前は前処理を別のカーネル（qk_norm_rope_cache）にしていましたが、1 トークンに 11 回の起動と、キャッシュに
// 残っていない重みの読み出しの待ちが毎回かかるため、ここにまとめました。
//
// 分担: スレッドブロック (g, s) が、KV ヘッド g と、それを共有する kAttentionGroup 個の query ヘッドについて、
// 見る位置の範囲を「使う分割数」等分したうちの s 番目を受け持ちます（KV キャッシュを 1 回読めば 4 ヘッド分に使える）。
// 使う分割数は、1 分割が kAttentionMinPositionsPerSplit 位置以上になるように、範囲の長さから決めます（最大 gridDim.y）。
// 使わないブロックはすぐに終わります。
// 範囲の中は kAttentionChunk（64）位置ずつ、次の 3 段で処理し、最大値・合計・重み付き和を逐次更新します（online softmax）。
//   1. score: 16 レーンで 1 位置を受け持ち（1 レーン 8 次元）、16 位置 × 4 回で 64 位置。K と V（段 3 で使う）を
//      最初にまとめて読み、4 ヘッド分の内積を共有メモリに置く
//   2. 重み: ワープ h がヘッド h の最大値を求め、exp(score - 最大値) に置き換えて合計する
//   3. 重み付き和: 段 1 で読んだ V（同じスレッドが同じ位置を読んでいる）に重みを掛け、4 ヘッド分まとめて足す
// 最後に 16 組の重み付き和を足し合わせます。使う分割数が 1 なら、合計で割ってゲートを掛けて出力します。
// 2 以上なら、各ブロックは結果を partial に書いて done_count[g] を 1 増やし、最後に終わったブロックが
// 全ブロックの結果を合成して出力し、done_count[g] を 0 に戻します（次の起動のため）。出力は int8 化します。
//
// window が 0 なら全文脈、正なら「現在の位置 - 位置 < window」の位置だけを見ます（Hugging Face と同じ）。
// 起動: grid = (num_kv_heads, 最大の分割数)、block = 256。head_dim は 128、num_heads = 4 × num_kv_heads に限ります。
// partial は num_kv_heads × 最大の分割数 × 4 × 130 個の float、done_count は num_kv_heads 個の 0 で始まる int です。
// 例: 位置 300 の全文脈（301 位置）なら使う分割数は 3（1 ブロック 101 位置）、位置 100 なら 1
// 1 分割の最小を 128 位置にしたのは、合成（部分結果の書き込み・atomic・読み直し）に約 2 µs かかり、短い範囲では
// 分けない方が速かったためです（単独の計測で、位置 64 は分けると 7.8 µs、分けないと 6.9 µs）。
// 引数の例: qkv は q・k・v の GEMV の出力 [num_heads×128 | num_kv×128 | num_kv×128]、normed は層の入口の正規化の出力
// （ゲートの入力）、hidden_size = 2048、inverse_frequency は 64 個（head_dim / 2）。
__global__ void attention_decode(const float* __restrict__ qkv, const float* __restrict__ q_norm,
                                 const float* __restrict__ k_norm, const float* __restrict__ inverse_frequency,
                                 const __half* __restrict__ gate_weight, const float* __restrict__ normed,
                                 int hidden_size, float eps, __half* __restrict__ key_cache,
                                 __half* __restrict__ value_cache, const int* __restrict__ state, int window,
                                 int num_kv_heads, float scale, float* __restrict__ partial,
                                 int* __restrict__ done_count, int8_t* __restrict__ q,
                                 ActBlockInfo* __restrict__ info) {
    __shared__ float query_shared[kAttentionGroup][kAttentionHeadDim];
    __shared__ float key_shared[kAttentionHeadDim];  // 正規化した後、RoPE の前の k
    __shared__ __align__(16) __half current_key[kAttentionHeadDim];    // 現在の位置の k（RoPE の後、FP16）
    __shared__ __align__(16) __half current_value[kAttentionHeadDim];  // 現在の位置の v（FP16）
    __shared__ float prologue_query_squares[kAttentionWarps];  // ワープごとの q の 2 乗和
    __shared__ float prologue_key_squares[kAttentionWarps];    // ワープごとの k の 2 乗和
    __shared__ float prologue_gate[kAttentionWarps][kAttentionGroup];  // ワープごとのゲートの内積
    __shared__ float gate_shared[kAttentionGroup];
    __shared__ float weights[kAttentionGroup][kAttentionChunk];
    __shared__ float correction_shared[kAttentionGroup];
    __shared__ float warp_sums[kAttentionWarps][kAttentionGroup][kAttentionHeadDim];
    __shared__ float head_max[kAttentionGroup];
    __shared__ float head_total[kAttentionGroup];
    __shared__ float result[kAttentionGroup][kAttentionHeadDim];
    __shared__ bool is_last_block;

    const int kv_head = blockIdx.x;
    const int split = blockIdx.y;
    const int thread = threadIdx.x;
    const int lane = thread % kWarpSize;
    const int warp = thread / kWarpSize;
    const int first_head = kv_head * kAttentionGroup;
    const int kv_width = num_kv_heads * kAttentionHeadDim;

    const int num_heads = kAttentionGroup * num_kv_heads;

    // 前処理の分担: スレッドは q のヘッド head の次元 dim0, dim0+1 を受け持つ。最初の 64 スレッドは k・v の
    // 同じ次元も受け持つ。値は直前の GEMV の後でキャッシュに残っていないものが多いので、使う値を最初にまとめて読む。
    const int head = thread / kPrologueThreadsPerHead;
    const int dim0 = (thread % kPrologueThreadsPerHead) * kPrologueDims;
    const bool has_key = thread < kPrologueThreadsPerHead;
    const float2 zero2 = make_float2(0.0f, 0.0f);
    const float* key_source = qkv + num_heads * kAttentionHeadDim + kv_head * kAttentionHeadDim + dim0;
    const float2 query_raw =
        *reinterpret_cast<const float2*>(qkv + (first_head + head) * kAttentionHeadDim + dim0);
    const float2 key_raw = has_key ? *reinterpret_cast<const float2*>(key_source) : zero2;
    const float2 value_raw = has_key ? *reinterpret_cast<const float2*>(key_source + kv_width) : zero2;
    const float2 query_weight = *reinterpret_cast<const float2*>(q_norm + dim0);
    const float2 key_weight = has_key ? *reinterpret_cast<const float2*>(k_norm + dim0) : zero2;
    const float2 frequency = *reinterpret_cast<const float2*>(inverse_frequency + dim0 % (kAttentionHeadDim / 2));
    const int position = state[1];
    // ゲートの内積（4 ヘッド × hidden）。展開して、複数回分の読み出しを同時に出す。
    float gate_dot[kAttentionGroup] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 8
    for (int i = thread; i < hidden_size; i += kAttentionThreads) {
        const float x = normed[i];
#pragma unroll
        for (int h = 0; h < kAttentionGroup; ++h) {
            gate_dot[h] += __half2float(gate_weight[static_cast<size_t>(first_head + h) * hidden_size + i]) * x;
        }
    }

    // 使う分割数と、このブロックが受け持つ位置の範囲 [begin, end)。
    const int first = window > 0 ? max(0, position - window + 1) : 0;
    const int length = position - first + 1;
    const int splits = min(static_cast<int>(gridDim.y),
                           (length + kAttentionMinPositionsPerSplit - 1) / kAttentionMinPositionsPerSplit);
    if (split >= splits) return;  // ブロック全体で同じ条件なので、__syncthreads の前に抜けてよい
    const int per_split = (length + splits - 1) / splits;
    const int begin = first + split * per_split;
    const int end = min(begin + per_split, position + 1);

    // 2 乗和とゲートの内積を、ワープごとに足してから共有メモリで合わせる。
    // q のヘッド head は 2 ワープ（kPrologueWarpsPerHead）、k は最初の 2 ワープが受け持つ。
    {
        const float query_squares = warp_sum(query_raw.x * query_raw.x + query_raw.y * query_raw.y);
        const float key_squares = warp_sum(key_raw.x * key_raw.x + key_raw.y * key_raw.y);
#pragma unroll
        for (int h = 0; h < kAttentionGroup; ++h) gate_dot[h] = warp_sum(gate_dot[h]);
        if (lane == 0) {
            prologue_query_squares[warp] = query_squares;
            prologue_key_squares[warp] = key_squares;
#pragma unroll
            for (int h = 0; h < kAttentionGroup; ++h) prologue_gate[warp][h] = gate_dot[h];
        }
    }
    __syncthreads();
    float query_total = 0.0f;
    float key_total = 0.0f;
    for (int w = 0; w < kPrologueWarpsPerHead; ++w) {
        query_total += prologue_query_squares[head * kPrologueWarpsPerHead + w];
        key_total += prologue_key_squares[w];
    }
    const float query_inverse_rms = 1.0f / sqrtf(query_total / kAttentionHeadDim + eps);
    const float key_inverse_rms = 1.0f / sqrtf(key_total / kAttentionHeadDim + eps);
    query_shared[head][dim0] = query_weight.x * (query_raw.x * query_inverse_rms);
    query_shared[head][dim0 + 1] = query_weight.y * (query_raw.y * query_inverse_rms);
    if (has_key) {
        key_shared[dim0] = key_weight.x * (key_raw.x * key_inverse_rms);
        key_shared[dim0 + 1] = key_weight.y * (key_raw.y * key_inverse_rms);
    }
    if (thread < kAttentionGroup) {
        float dot = 0.0f;
        for (int w = 0; w < kAttentionWarps; ++w) dot += prologue_gate[w][thread];
        gate_shared[thread] = sigmoid(dot);
    }
    __syncthreads();

    // RoPE。組になる次元は別のスレッドが持つので、全員が回転前の値を読んでから上書きする。
    const float angle0 = static_cast<float>(position) * frequency.x;
    const float angle1 = static_cast<float>(position) * frequency.y;
    const float cosine0 = cosf(angle0);
    const float sine0 = sinf(angle0);
    const float cosine1 = cosf(angle1);
    const float sine1 = sinf(angle1);
    const float query_rotated0 = rotate_half_at(query_shared[head], dim0, cosine0, sine0);
    const float query_rotated1 = rotate_half_at(query_shared[head], dim0 + 1, cosine1, sine1);
    const float key_rotated0 = has_key ? rotate_half_at(key_shared, dim0, cosine0, sine0) : 0.0f;
    const float key_rotated1 = has_key ? rotate_half_at(key_shared, dim0 + 1, cosine1, sine1) : 0.0f;
    __syncthreads();
    query_shared[head][dim0] = query_rotated0;
    query_shared[head][dim0 + 1] = query_rotated1;
    if (has_key) {
        const __half2 key_half = __floats2half2_rn(key_rotated0, key_rotated1);
        const __half2 value_half = __floats2half2_rn(value_raw.x, value_raw.y);
        *reinterpret_cast<__half2*>(current_key + dim0) = key_half;
        *reinterpret_cast<__half2*>(current_value + dim0) = value_half;
        // 現在の位置を受け持つブロック（範囲の最後）だけが、KV キャッシュに書く（次のトークンから使う）。
        if (end == position + 1) {
            const size_t slot = static_cast<size_t>(position) * kv_width + kv_head * kAttentionHeadDim + dim0;
            *reinterpret_cast<__half2*>(key_cache + slot) = key_half;
            *reinterpret_cast<__half2*>(value_cache + slot) = value_half;
        }
    }
    __syncthreads();  // query_shared・current_key・current_value・gate_shared の書き込みを待つ

    // 段 1・3 の分担: 16 レーンの組（slot、0〜15）が 1 位置を受け持ち、column が次元 column×8 から 8 個を受け持つ。
    // 共有メモリから毎回読むと、16 レーンが 8 要素おきの番地を読んでバンクが衝突するため、query はレジスタに置く。
    const int column = thread % kLanesPerPosition;
    const int slot = thread / kLanesPerPosition;
    float query_part[kAttentionGroup][kDimsPerScoreLane];
#pragma unroll
    for (int h = 0; h < kAttentionGroup; ++h) {
#pragma unroll
        for (int j = 0; j < kDimsPerScoreLane; ++j) query_part[h][j] = query_shared[h][column * kDimsPerScoreLane + j];
    }
    float acc[kAttentionGroup][kDimsPerScoreLane];
#pragma unroll
    for (int h = 0; h < kAttentionGroup; ++h) {
#pragma unroll
        for (int j = 0; j < kDimsPerScoreLane; ++j) acc[h][j] = 0.0f;
    }
    // 段 2 の状態（ワープ h < 4 がヘッド h について持つ。ワープ内の全レーンで同じ値）。
    float running_max = -INFINITY;
    float running_total = 0.0f;

    for (int chunk = begin; chunk < end; chunk += kAttentionChunk) {
        const int count = min(kAttentionChunk, end - chunk);
        const size_t chunk_offset =
            static_cast<size_t>(chunk) * kv_width + kv_head * kAttentionHeadDim + column * kDimsPerScoreLane;
        // 段 1: K と V をまとめて読む（範囲外は 0）。読み出しの待ちは 1 回分で済む。
        uint4 keys[kPassesPerChunk];
        uint4 values[kPassesPerChunk];
#pragma unroll
        for (int pass = 0; pass < kPassesPerChunk; ++pass) {
            const int index = pass * kPositionsPerPass + slot;
            const size_t offset = chunk_offset + static_cast<size_t>(index) * kv_width;
            keys[pass] = make_uint4(0, 0, 0, 0);
            values[pass] = make_uint4(0, 0, 0, 0);
            if (index < count) {
                if (chunk + index == position) {
                    // 現在の位置は、このカーネルが書く前の値がキャッシュにあるため、共有メモリから使う。
                    keys[pass] = *reinterpret_cast<const uint4*>(current_key + column * kDimsPerScoreLane);
                    values[pass] = *reinterpret_cast<const uint4*>(current_value + column * kDimsPerScoreLane);
                } else {
                    keys[pass] = __ldg(reinterpret_cast<const uint4*>(key_cache + offset));
                    values[pass] = __ldg(reinterpret_cast<const uint4*>(value_cache + offset));
                }
            }
        }
#pragma unroll
        for (int pass = 0; pass < kPassesPerChunk; ++pass) {
            const int index = pass * kPositionsPerPass + slot;
            float key_part[kDimsPerScoreLane];
            unpack_halves(keys[pass], key_part);
            float dots[kAttentionGroup] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll
            for (int h = 0; h < kAttentionGroup; ++h) {
#pragma unroll
                for (int j = 0; j < kDimsPerScoreLane; ++j) dots[h] += query_part[h][j] * key_part[j];
                for (int offset = kLanesPerPosition / 2; offset > 0; offset /= 2) {
                    dots[h] += __shfl_xor_sync(kFullMask, dots[h], offset);
                }
            }
            if (index < count && column == 0) {
#pragma unroll
                for (int h = 0; h < kAttentionGroup; ++h) weights[h][index] = dots[h] * scale;
            }
        }
        __syncthreads();

        // 段 2: ワープ h がヘッド h の重みを求める（1 レーン 2 位置）。最大値の更新に合わせて、
        // これまでの合計と重み付き和を縮める係数 correction を、段 3 のために共有メモリに置く。
        if (warp < kAttentionGroup) {
            float local_max = -INFINITY;
            for (int i = lane; i < count; i += kWarpSize) local_max = fmaxf(local_max, weights[warp][i]);
            const float new_max = fmaxf(running_max, warp_max(local_max));
            const float correction = expf(running_max - new_max);  // 最初は exp(-inf) = 0
            float local_total = 0.0f;
            for (int i = lane; i < count; i += kWarpSize) {
                const float weight = expf(weights[warp][i] - new_max);
                weights[warp][i] = weight;
                local_total += weight;
            }
            running_total = running_total * correction + warp_sum(local_total);
            running_max = new_max;
            if (lane == 0) correction_shared[warp] = correction;
        }
        __syncthreads();

        // 段 3: 重み付き和。範囲外の位置の重みは 0 とする（共有メモリには前の区切りの値や、最初は不定の値が残っており、
        // NaN の可能性もあるため、V が 0 でも掛けずに 0 にする）。
#pragma unroll
        for (int h = 0; h < kAttentionGroup; ++h) {
            const float factor = correction_shared[h];
#pragma unroll
            for (int j = 0; j < kDimsPerScoreLane; ++j) acc[h][j] *= factor;
        }
#pragma unroll
        for (int pass = 0; pass < kPassesPerChunk; ++pass) {
            const int index = pass * kPositionsPerPass + slot;
            float value_part[kDimsPerScoreLane];
            unpack_halves(values[pass], value_part);
#pragma unroll
            for (int h = 0; h < kAttentionGroup; ++h) {
                const float weight = index < count ? weights[h][index] : 0.0f;
#pragma unroll
                for (int j = 0; j < kDimsPerScoreLane; ++j) acc[h][j] += weight * value_part[j];
            }
        }
        __syncthreads();  // 次の区切りが weights を上書きする前に、全員が読み終わるのを待つ
    }

    // 16 個の slot の重み付き和を足し合わせる。ワープの前半と後半（slot 2w と 2w+1）はシャッフルで、
    // ワープ間は共有メモリで足す。
#pragma unroll
    for (int h = 0; h < kAttentionGroup; ++h) {
#pragma unroll
        for (int j = 0; j < kDimsPerScoreLane; ++j) {
            acc[h][j] += __shfl_xor_sync(kFullMask, acc[h][j], kLanesPerPosition);
        }
    }
    if (lane < kLanesPerPosition) {
#pragma unroll
        for (int h = 0; h < kAttentionGroup; ++h) {
#pragma unroll
            for (int j = 0; j < kDimsPerScoreLane; ++j) warp_sums[warp][h][column * kDimsPerScoreLane + j] = acc[h][j];
        }
    }
    if (warp < kAttentionGroup && lane == 0) {
        head_max[warp] = running_max;
        head_total[warp] = running_total;
    }
    __syncthreads();

    if (splits == 1) {
        // 合計で割り、ゲートを掛けて result に置く。
        for (int i = thread; i < kAttentionGroup * kAttentionHeadDim; i += kAttentionThreads) {
            const int h = i / kAttentionHeadDim;
            const int d = i % kAttentionHeadDim;
            float sum = 0.0f;
            for (int w = 0; w < kAttentionWarps; ++w) sum += warp_sums[w][h][d];
            result[h][d] = sum / head_total[h] * gate_shared[h];
        }
    } else {
        // 部分結果を書き、最後に終わったブロックかどうかを確かめる。
        float* mine = partial + static_cast<size_t>(kv_head * gridDim.y + split) * kAttentionGroup * kAttentionPartialStride;
        for (int i = thread; i < kAttentionGroup * kAttentionHeadDim; i += kAttentionThreads) {
            const int h = i / kAttentionHeadDim;
            const int d = i % kAttentionHeadDim;
            float sum = 0.0f;
            for (int w = 0; w < kAttentionWarps; ++w) sum += warp_sums[w][h][d];
            mine[h * kAttentionPartialStride + 2 + d] = sum;
        }
        if (thread < kAttentionGroup) {
            mine[thread * kAttentionPartialStride] = head_max[thread];
            mine[thread * kAttentionPartialStride + 1] = head_total[thread];
        }
        __threadfence();  // 部分結果の書き込みを、done_count を増やす前に他のブロックから見えるようにする
        __syncthreads();
        if (thread == 0) is_last_block = atomicAdd(&done_count[kv_head], 1) == splits - 1;
        __syncthreads();
        if (!is_last_block) return;
        __threadfence();
        // 合成: 全ブロックの最大値に合わせて足し合わせる。他のブロックの書き込みを読むため、L1 を通らない __ldcg を使う。
        const float* group_partial =
            partial + static_cast<size_t>(kv_head * gridDim.y) * kAttentionGroup * kAttentionPartialStride;
        for (int i = thread; i < kAttentionGroup * kAttentionHeadDim; i += kAttentionThreads) {
            const int h = i / kAttentionHeadDim;
            const int d = i % kAttentionHeadDim;
            float overall_max = -INFINITY;
            for (int s = 0; s < splits; ++s) {
                overall_max = fmaxf(overall_max, __ldcg(group_partial + (s * kAttentionGroup + h) * kAttentionPartialStride));
            }
            float total = 0.0f;
            float sum = 0.0f;
            for (int s = 0; s < splits; ++s) {
                const float* part = group_partial + (s * kAttentionGroup + h) * kAttentionPartialStride;
                const float factor = expf(__ldcg(part) - overall_max);
                total += __ldcg(part + 1) * factor;
                sum += __ldcg(part + 2 + d) * factor;
            }
            result[h][d] = sum / total * gate_shared[h];
        }
        if (thread == 0) done_count[kv_head] = 0;
    }
    __syncthreads();

    // 4 ヘッド × 128 要素 = int8 化の 16 ブロック。8 ワープで 2 ブロックずつ担当する。
    const int blocks_per_head = kAttentionHeadDim / kQuantBlock;
    for (int block = warp; block < kAttentionGroup * blocks_per_head; block += kAttentionWarps) {
        const int local_head = block / blocks_per_head;
        const int part = block % blocks_per_head;
        quantize_warp(result[local_head][part * kQuantBlock + lane], (first_head + local_head) * blocks_per_head + part,
                      q, info);
    }
}

// silu_mul_quantize_at は、FFN の中間値 a[index] = silu(gate[index]) × up[index] を計算して int8 化します。
// gate_up は [gate（n 個）, up（n 個）] の 1 トークン分です。ワープの全レーンで呼びます（n は 32 の倍数）。
__device__ __forceinline__ void silu_mul_quantize_at(const float* __restrict__ gate_up, int n, int index,
                                                     int8_t* __restrict__ q, ActBlockInfo* __restrict__ info) {
    const float g = gate_up[index];
    const float activated = g * sigmoid(g) * gate_up[n + index];
    quantize_warp(activated, index / kQuantBlock, q, info);
}

// silu_mul_quantize は、FFN の中間値 a = silu(gate) × up を計算して int8 化します（down の入力）。
// gate_up は [gate（n 個）, up（n 個）] の順に並んだ GEMV の出力です。
// 起動: grid = n / 256、block = 256。例: n = 7680 なら grid = 30
__global__ void silu_mul_quantize(const float* __restrict__ gate_up, int n, int8_t* __restrict__ q,
                                  ActBlockInfo* __restrict__ info) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;  // n は 32 の倍数なので、ワープ単位で抜ける
    silu_mul_quantize_at(gate_up, n, index, q, info);
}

// argmax_partial は、logits を grid のスレッドブロック数に分け、各部分の最大値とその位置を求めます。
// 同じ値なら位置の小さい方を選びます（CPU 版の argmax と同じ）。
// 起動: grid = parts（例: 256）、block = 256
__global__ void argmax_partial(const float* __restrict__ logits, int n, float* __restrict__ part_value,
                               int* __restrict__ part_index) {
    __shared__ float best_values[256];
    __shared__ int best_indices[256];
    float best = -INFINITY;
    int best_index = n;  // 見つからなかった印（どの有効な位置より大きい）
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) {
        const float value = logits[i];
        if (value > best) {
            best = value;
            best_index = i;
        }
    }
    best_values[threadIdx.x] = best;
    best_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float other = best_values[threadIdx.x + stride];
            const int other_index = best_indices[threadIdx.x + stride];
            if (other > best_values[threadIdx.x] ||
                (other == best_values[threadIdx.x] && other_index < best_indices[threadIdx.x])) {
                best_values[threadIdx.x] = other;
                best_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        part_value[blockIdx.x] = best_values[0];
        part_index[blockIdx.x] = best_indices[0];
    }
}

// argmax_final_advance は、argmax_partial の部分結果から全体の argmax を求め、
// state[2]（結果）と state[0]（次の入力トークン）に書き、state[1]（位置）を 1 進めます。
// 貪欲生成では、ホストが state を書き換えずに次の Graph を再生できます。
// 起動: grid = 1、block = 256（parts ≤ 256）
__global__ void argmax_final_advance(const float* __restrict__ part_value, const int* __restrict__ part_index,
                                     int parts, int* __restrict__ state) {
    __shared__ float best_values[256];
    __shared__ int best_indices[256];
    const bool valid = threadIdx.x < parts;
    best_values[threadIdx.x] = valid ? part_value[threadIdx.x] : -INFINITY;
    best_indices[threadIdx.x] = valid ? part_index[threadIdx.x] : INT32_MAX;
    __syncthreads();
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2) {
        if (threadIdx.x < stride) {
            const float other = best_values[threadIdx.x + stride];
            const int other_index = best_indices[threadIdx.x + stride];
            if (other > best_values[threadIdx.x] ||
                (other == best_values[threadIdx.x] && other_index < best_indices[threadIdx.x])) {
                best_values[threadIdx.x] = other;
                best_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        state[2] = best_indices[0];
        state[0] = best_indices[0];
        state[1] = state[1] + 1;
    }
}

// candidates_partial は、選び方の候補を絞る 1 段目です。番号 [0, valid_tokens) を grid のブロック数（64）の区間に分け、
// 1 ブロックが 1 区間について次を行います。
//   1. accept_token（直前に生成して今回入力したトークン。なければ -1）がこの区間なら、seen と counts に記録する
//      （valid_tokens 以上ならブロック 0 が記録する）
//   2. ペナルティを適用した値の最大値と exp((値 - 最大値) × inverse_temperature) の合計を求める
//   3. 各ワープが kCandidateCount 個を並べ、ワープ間でまとめて、区間の上位 kCandidateCount 個を part_values・part_ids に書く
// 起動: grid = 64、block = kCandidatePartThreads（512）。
// 1 区間の長さは kCandidatePartWarps × kCandidateCount 以下であること（valid_tokens ≤ 64 × 2048。LFM3 は 125,033）。
__global__ void candidates_partial(const float* __restrict__ logits, int valid_tokens,
                                   unsigned char* __restrict__ seen, int* __restrict__ counts, int accept_token,
                                   float inverse_temperature, float repetition_penalty, float presence_penalty,
                                   float frequency_penalty, float* __restrict__ part_values,
                                   int* __restrict__ part_ids, float* __restrict__ part_max,
                                   float* __restrict__ part_sum) {
    __shared__ Candidate lists[kCandidatePartWarps / 2 * kCandidateCount];
    __shared__ float scratch[kWarpSize];
    const int span = (valid_tokens + gridDim.x - 1) / gridDim.x;
    const int begin = blockIdx.x * span;
    const int end = min(begin + span, valid_tokens);
    // 記録は区間を担当するブロックだけが行い、同じブロックが下で読むため、ブロック内の同期で十分です。
    // valid_tokens 以上の番号（どの区間にも入らない）は、今回はどのブロックも読まないため、ブロック 0 が記録します。
    const bool owns_accept_token = (accept_token >= begin && accept_token < end) ||
                                   (blockIdx.x == 0 && accept_token >= valid_tokens);
    if (threadIdx.x == 0 && owns_accept_token) {
        seen[accept_token] = 1;
        counts[accept_token] += 1;
    }
    __syncthreads();

    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    Candidate items[kCandidatesPerLane];
    float local_max = -INFINITY;
#pragma unroll
    for (int j = 0; j < kCandidatesPerLane; ++j) {
        const int id = begin + warp * kCandidateCount + lane * kCandidatesPerLane + j;
        items[j] = {-INFINITY, kCandidateMissingId};
        if (id < end) {
            items[j] = {penalized_logit(__ldg(&logits[id]), seen[id] != 0, counts[id], repetition_penalty,
                                        presence_penalty, frequency_penalty),
                        id};
        }
        local_max = fmaxf(local_max, items[j].value);
    }
    const float block_max = block_reduce_max(local_max, scratch);
    float local_sum = 0.0f;
#pragma unroll
    for (int j = 0; j < kCandidatesPerLane; ++j) {
        if (items[j].id != kCandidateMissingId) {
            local_sum += expf((items[j].value - block_max) * inverse_temperature);
        }
    }
    const float block_sum = block_reduce_sum(local_sum, scratch);

    warp_sort_candidates(items);
    merge_warp_candidates(items, lists, kCandidatePartWarps);
    if (warp == 0) {
#pragma unroll
        for (int j = 0; j < kCandidatesPerLane; ++j) {
            const int index = blockIdx.x * kCandidateCount + lane * kCandidatesPerLane + j;
            part_values[index] = items[j].value;
            part_ids[index] = items[j].id;
        }
    }
    if (threadIdx.x == 0) {
        part_max[blockIdx.x] = block_max;
        part_sum[blockIdx.x] = block_sum;
    }
}

// candidates_final は、選び方の候補を絞る 2 段目です。1 段目の 64 本の列（各 kCandidateCount 個、
// 最良が先の順）から全体の上位 kCandidateCount 個を out_values・out_ids に書き、out_stats に
// [全体の最大値, 全体の exp の合計] を書きます。ワープ w は列 w と列 w + kCandidateFinalWarps をまとめてから、
// ワープ間でまとめます。exp の合計は、各区間の合計を全体の最大値に合わせて足したもの
// （Σ part_sum × exp((part_max - 最大値) × inverse_temperature)）です。
// 起動: grid = 1、block = kCandidateFinalThreads（1024）。parts は 2 × kCandidateFinalWarps（64）であること。
__global__ void candidates_final(const float* __restrict__ part_values, const int* __restrict__ part_ids,
                                 const float* __restrict__ part_max, const float* __restrict__ part_sum, int parts,
                                 float inverse_temperature, float* __restrict__ out_values,
                                 int* __restrict__ out_ids, float* __restrict__ out_stats) {
    __shared__ Candidate lists[kCandidateFinalWarps * kCandidateCount];
    const int warp = threadIdx.x / kWarpSize;
    const int lane = threadIdx.x % kWarpSize;
    // ワープ w の 2 本目の列（列 w + 32）を共有メモリに置き、1 本目（列 w）をレジスタに読んでまとめます。
    for (int i = lane; i < kCandidateCount; i += kWarpSize) {
        const int source = (warp + kCandidateFinalWarps) * kCandidateCount + i;
        lists[warp * kCandidateCount + i] = {part_values[source], part_ids[source]};
    }
    Candidate items[kCandidatesPerLane];
#pragma unroll
    for (int j = 0; j < kCandidatesPerLane; ++j) {
        const int source = warp * kCandidateCount + lane * kCandidatesPerLane + j;
        items[j] = {part_values[source], part_ids[source]};
    }
    __syncwarp();
    warp_merge_candidates(items, lists + warp * kCandidateCount);
    __syncthreads();  // lists をワープ間のまとめに使う前に、全ワープが読み終えるのを待つ
    merge_warp_candidates(items, lists, kCandidateFinalWarps);
    if (warp == 0) {
#pragma unroll
        for (int j = 0; j < kCandidatesPerLane; ++j) {
            out_values[lane * kCandidatesPerLane + j] = items[j].value;
            out_ids[lane * kCandidatesPerLane + j] = items[j].id;
        }
    }
    if (threadIdx.x == 0) {
        float overall_max = -INFINITY;
        for (int part = 0; part < parts; ++part) {
            overall_max = fmaxf(overall_max, part_max[part]);
        }
        float overall_sum = 0.0f;
        for (int part = 0; part < parts; ++part) {
            if (part_sum[part] > 0.0f) {
                overall_sum += part_sum[part] * expf((part_max[part] - overall_max) * inverse_temperature);
            }
        }
        out_stats[0] = overall_max;
        out_stats[1] = overall_sum;
    }
}


// ===== 入力処理（prefill、M4）=====
// Prefill は最大 512 トークンを 1 区切りとして、各カーネルがトークンの行をまとめて処理します。
// 行のデータはトークンごとに連続して並びます（例: hidden は [トークン][hidden]）。int8 化した活性値は、
// 次の行列の入力次元 n ごとに [トークン][n]（int8）と [トークン][n / 32]（ActBlockInfo）です。

// embed_rows は、tokens[t] の埋め込み（FP16）を FP32 にして hidden の行 t に書きます。
// 起動: grid = (hidden / 256, トークン数)、block = 256
__global__ void embed_rows(const __half* __restrict__ table, const int* __restrict__ tokens,
                           float* __restrict__ hidden, int hidden_size) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_size) return;
    const int row = blockIdx.y;
    const int token = tokens[row];
    hidden[static_cast<size_t>(row) * hidden_size + index] =
        __half2float(table[static_cast<size_t>(token) * hidden_size + index]);
}

// residual_norm_quantize_rows は、residual_norm_quantize をトークンの行ごとに行います（ブロック t が行 t）。
// delta が null なら加算をしません。起動: grid = トークン数、block = 1024
__global__ void residual_norm_quantize_rows(float* __restrict__ hidden, const float* __restrict__ delta,
                                            const float* __restrict__ delta_weight,
                                            const float* __restrict__ next_weight, float eps, int n,
                                            float* __restrict__ normed, int8_t* __restrict__ q,
                                            ActBlockInfo* __restrict__ info) {
    const size_t row = blockIdx.x;
    residual_norm_quantize_vector(hidden + row * n, delta != nullptr ? delta + row * n : nullptr, delta_weight,
                                  next_weight, eps, n, normed + row * n, q + row * n, info + row * (n / kQuantBlock));
}

// silu_mul_quantize_rows は、silu_mul_quantize をトークンの行ごとに行います。gate_up は [トークン][2 × n] です。
// 起動: grid = (n / 256, トークン数)、block = 256
__global__ void silu_mul_quantize_rows(const float* __restrict__ gate_up, int n, int8_t* __restrict__ q,
                                       ActBlockInfo* __restrict__ info) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    const size_t row = blockIdx.y;
    silu_mul_quantize_at(gate_up + row * 2 * n, n, index, q + row * n, info + row * (n / kQuantBlock));
}

// quantize_rows は、FP32 の行 [トークン][n] を 32 要素ごとに int8 化します（attention の出力 → o_proj の入力）。
// 起動: grid = (n / 256, トークン数)、block = 256
__global__ void quantize_rows(const float* __restrict__ values, int n, int8_t* __restrict__ q,
                              ActBlockInfo* __restrict__ info) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    const size_t row = blockIdx.y;
    quantize_warp(values[row * n + index], index / kQuantBlock, q + row * n, info + row * (n / kQuantBlock));
}

}  // extern "C"

// gemm_pq4_tile は PQ4/PQ8 の重みと int8 活性値のブロック積を FP32 で同じ順に累積する。
// 例として基準形状は64行×128トークン、Gemma 4の入力処理では128行×64トークンを使う。
// 行数・重みのブロック数はそれぞれタイル行数・K方向の段幅で割り切れる形状を渡す。
template <int kGemmTileRows, int kGemmTileTokens, int kGemmThreadRows, int kGemmThreadTokens, int kGemmStageBlocks>
__device__ __forceinline__ void gemm_pq4_tile(const int4* __restrict__ packed,
                                                         const __half* __restrict__ scales,
                                                         const int8_t* __restrict__ act_q,
                                                         const ActBlockInfo* __restrict__ act_info,
                                                         float* __restrict__ out, int rows, int blocks_per_row,
                                                         int tokens, int input_block_shift) {
    constexpr int rowGroups = kGemmTileRows / kGemmThreadRows;
    static_assert(rowGroups == 16, "16 row groups and 16 token groups");
    static_assert(kGemmThreadTokens * 16 == kGemmTileTokens, "token tile size");
    __shared__ int4 weight_tile[kGemmStageBlocks][kGemmTileRows];
    __shared__ float scale_tile[kGemmStageBlocks][kGemmTileRows];
    __shared__ int4 act_tile[kGemmStageBlocks][kGemmTileTokens][2];
    __shared__ ActBlockInfo info_tile[kGemmStageBlocks][kGemmTileTokens];

    const int thread = threadIdx.x;
    const int rx = thread % rowGroups;
    const int ty = thread / rowGroups;
    const int first_row = blockIdx.x * kGemmTileRows;
    const int first_token = blockIdx.y * kGemmTileTokens;
    const int4* act_vectors = reinterpret_cast<const int4*>(act_q);  // 1 ブロック = int4 2 個（32 バイト）
    const int input_blocks_per_row = blocks_per_row >> input_block_shift;  // 活性値の 1 トークンあたりのブロック数

    float acc[kGemmThreadRows][kGemmThreadTokens];
#pragma unroll
    for (int i = 0; i < kGemmThreadRows; ++i) {
#pragma unroll
        for (int j = 0; j < kGemmThreadTokens; ++j) acc[i][j] = 0.0f;
    }

    for (int stage = 0; stage < blocks_per_row; stage += kGemmStageBlocks) {
        // 重みと係数: kGemmTileRows 行 × kGemmStageBlocks ブロック。連続した番号のスレッドが、同じ行の連続したブロックを読む。
        for (int i = thread; i < kGemmTileRows * kGemmStageBlocks; i += kGemmThreads) {
            const int row = i / kGemmStageBlocks;
            const int block = i % kGemmStageBlocks;
            const size_t offset = static_cast<size_t>(first_row + row) * blocks_per_row + stage + block;
            weight_tile[block][row] = packed[offset];
            scale_tile[block][row] = __half2float(scales[offset]);
        }
        // 活性値と付帯情報: kGemmTileTokens トークン × kGemmStageBlocks ブロック。範囲外のトークンは 0 にする（結果は書かない）。
        for (int i = thread; i < kGemmTileTokens * kGemmStageBlocks * 2; i += kGemmThreads) {
            const int token = i / (kGemmStageBlocks * 2);
            const int block = (i / 2) % kGemmStageBlocks;
            const int half = i % 2;
            const int global_token = first_token + token;
            int4 value = make_int4(0, 0, 0, 0);
            if (global_token < tokens) {
                const int input_block = (stage + block) >> input_block_shift;
                value = act_vectors[(static_cast<size_t>(global_token) * input_blocks_per_row + input_block) * 2 + half];
            }
            act_tile[block][token][half] = value;
        }
        for (int i = thread; i < kGemmTileTokens * kGemmStageBlocks; i += kGemmThreads) {
            const int token = i / kGemmStageBlocks;
            const int block = i % kGemmStageBlocks;
            const int global_token = first_token + token;
            // 共有メモリには (scale, −8·scaled_sum) を置く（計算で 1 ブロックごとに 1 回の積和で済ませるため）。
            ActBlockInfo value = {0.0f, 0.0f};
            if (global_token < tokens) {
                const int input_block = (stage + block) >> input_block_shift;
                value = act_info[static_cast<size_t>(global_token) * input_blocks_per_row + input_block];
            }
            info_tile[block][token] = {value.scale, -kWeightCenter * value.scaled_sum};
        }
        __syncthreads();

#pragma unroll 2
        for (int block = 0; block < kGemmStageBlocks; ++block) {
            int low[kGemmThreadRows][4];
            int high[kGemmThreadRows][4];
            float weight_scale[kGemmThreadRows];
#pragma unroll
            for (int i = 0; i < kGemmThreadRows; ++i) {
                const int4 w = weight_tile[block][rx + i * rowGroups];
                low[i][0] = low_nibbles(w.x);
                low[i][1] = low_nibbles(w.y);
                low[i][2] = low_nibbles(w.z);
                low[i][3] = low_nibbles(w.w);
                high[i][0] = high_nibbles(w.x);
                high[i][1] = high_nibbles(w.y);
                high[i][2] = high_nibbles(w.z);
                high[i][3] = high_nibbles(w.w);
                weight_scale[i] = scale_tile[block][rx + i * rowGroups];
            }
#pragma unroll
            for (int j = 0; j < kGemmThreadTokens; ++j) {
                const int token = ty + j * rowGroups;
                const int4 a0 = act_tile[block][token][0];
                const int4 a1 = act_tile[block][token][1];
                const ActBlockInfo info = info_tile[block][token];
#pragma unroll
                for (int i = 0; i < kGemmThreadRows; ++i) {
                    int dot = 0;
                    dot = __dp4a(low[i][0], a0.x, dot);
                    dot = __dp4a(low[i][1], a0.y, dot);
                    dot = __dp4a(low[i][2], a0.z, dot);
                    dot = __dp4a(low[i][3], a0.w, dot);
                    dot = __dp4a(high[i][0], a1.x, dot);
                    dot = __dp4a(high[i][1], a1.y, dot);
                    dot = __dp4a(high[i][2], a1.z, dot);
                    dot = __dp4a(high[i][3], a1.w, dot);
                    // info.scaled_sum には −8·scaled_sum が入っている（読み込み時に計算済み）。
                    acc[i][j] += weight_scale[i] * fmaf(info.scale, static_cast<float>(dot), info.scaled_sum);
                }
            }
        }
        __syncthreads();  // 次の段で共有メモリを書き換える前に、全員が読み終わるのを待つ
    }

#pragma unroll
    for (int j = 0; j < kGemmThreadTokens; ++j) {
        const int token = first_token + ty + j * rowGroups;
        if (token >= tokens) continue;
#pragma unroll
        for (int i = 0; i < kGemmThreadRows; ++i) {
            out[static_cast<size_t>(token) * rows + first_row + rx + i * rowGroups] = acc[i][j];
        }
    }
}

extern "C" {
// 既存の64行×128トークン形状を同じABIで呼ぶ。
__global__ void __launch_bounds__(kGemmThreads) gemm_pq4(const int4* __restrict__ packed,
                                                         const __half* __restrict__ scales,
                                                         const int8_t* __restrict__ act_q,
                                                         const ActBlockInfo* __restrict__ act_info,
                                                         float* __restrict__ out, int rows, int blocks_per_row,
                                                         int tokens, int input_block_shift) {
    gemm_pq4_tile<kGemmTileRows, kGemmTileTokens, kGemmThreadRows, kGemmThreadTokens, kGemmStageBlocks>(packed, scales, act_q, act_info, out, rows, blocks_per_row, tokens, input_block_shift);
}

// short_conv_quantize_rows は、short_conv_quantize を区切りのトークン（位置 start_position から tokens 個）に
// 順に行います。1 スレッドが 1 チャネルを受け持ち、直近 K-1 個の u をレジスタに持って時間方向に進みます。
// 足す順・式は short_conv_quantize と同じです。最後に、輪状バッファ history を、1 個ずつ処理した場合と同じ状態
// （位置 p の u が行 p mod (K-1)）にします。conv_in は [トークン][3 × hidden] です。
// 起動: grid = hidden / 256、block = 256
__global__ void short_conv_quantize_rows(const float* __restrict__ conv_in, float* __restrict__ history,
                                         const float* __restrict__ kernel, int start_position, int tokens,
                                         int hidden_size, int kernel_size, int8_t* __restrict__ q,
                                         ActBlockInfo* __restrict__ info) {
    const int channel = blockIdx.x * blockDim.x + threadIdx.x;
    if (channel >= hidden_size) return;  // hidden_size は 32 の倍数なので、ワープ単位で抜ける
    const int history_length = kernel_size - 1;
    float past_taps[kConvMaxKernel - 1];
    float past[kConvMaxKernel - 1];  // past[k] は位置 p-(K-1)+k の u（p は次に処理する位置）
#pragma unroll
    for (int k = 0; k < kConvMaxKernel - 1; ++k) {
        const bool used = k < history_length;
        const int row = (start_position + k) % history_length;
        past_taps[k] = used ? kernel[static_cast<size_t>(k) * hidden_size + channel] : 0.0f;
        past[k] = used ? history[static_cast<size_t>(row) * hidden_size + channel] : 0.0f;
    }
    const float current_tap = kernel[static_cast<size_t>(history_length) * hidden_size + channel];
    const int blocks_per_row = hidden_size / kQuantBlock;
#pragma unroll 4
    for (int t = 0; t < tokens; ++t) {
        const float* row_in = conv_in + static_cast<size_t>(t) * 3 * hidden_size;
        const float current = row_in[channel] * row_in[2 * hidden_size + channel];
        float sum = current_tap * current;
#pragma unroll
        for (int k = 0; k < kConvMaxKernel - 1; ++k) sum += past_taps[k] * past[k];
        quantize_warp(row_in[hidden_size + channel] * sum, channel / kQuantBlock,
                      q + static_cast<size_t>(t) * hidden_size, info + static_cast<size_t>(t) * blocks_per_row);
        // 1 つ古い方へずらし、最後（k = K-2）に現在の u を入れる。添字を定数にするため、全 k を条件で書く。
#pragma unroll
        for (int k = 0; k < kConvMaxKernel - 1; ++k) {
            const int next_index = min(k + 1, kConvMaxKernel - 2);
            float next = 0.0f;
            if (k + 1 < history_length) next = past[next_index];
            if (k + 1 == history_length) next = current;
            past[k] = next;
        }
    }
    const int end_position = start_position + tokens;
#pragma unroll
    for (int k = 0; k < kConvMaxKernel - 1; ++k) {
        if (k < history_length) {
            history[static_cast<size_t>((end_position + k) % history_length) * hidden_size + channel] = past[k];
        }
    }
}

// attention_prepare_rows は、attention の前処理を 1 トークン 1 ブロックで行います（attention_decode の前処理と同じ式）。
//   - q = RMSNorm(q) × q_norm に RoPE を掛け、query の行 t に FP32 で書く（ヘッドごと）
//   - k = RMSNorm(k) × k_norm に RoPE を掛け、v はそのまま、FP16 で KV キャッシュの位置 start_position + t に書く
//   - ヘッドごとのゲート sigmoid(gate_weight[h] · normed[t]) を gate の行 t に書く
// 1 ワープが 1 ヘッドの正規化と RoPE を受け持ち、レーン l が次元 l, l+32, l+64, l+96 を持ちます（RoPE で組になる
// 次元 d と d+64 が同じレーンにあるため、シャッフルが要りません）。
// 起動: grid = トークン数、block = 256。qkv は [トークン][num_heads×128 | num_kv×128 | num_kv×128]。
__global__ void attention_prepare_rows(const float* __restrict__ qkv, const float* __restrict__ q_norm,
                                       const float* __restrict__ k_norm, const float* __restrict__ inverse_frequency,
                                       const __half* __restrict__ gate_weight, const float* __restrict__ normed,
                                       int hidden_size, float eps, __half* __restrict__ key_cache,
                                       __half* __restrict__ value_cache, int start_position, int num_kv_heads,
                                       float* __restrict__ query, float* __restrict__ gate) {
    constexpr int kDimsPerLane = kAttentionHeadDim / kWarpSize;  // 4
    const int row = blockIdx.x;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int warps = blockDim.x / kWarpSize;
    const int num_heads = kAttentionGroup * num_kv_heads;
    const int kv_width = num_kv_heads * kAttentionHeadDim;
    const int position = start_position + row;
    const float* row_qkv = qkv + static_cast<size_t>(row) * (num_heads * kAttentionHeadDim + 2 * kv_width);

    // 正規化と RoPE: ヘッド 0〜num_heads-1 が q、num_heads〜num_heads+num_kv_heads-1 が k。
    for (int head = warp; head < num_heads + num_kv_heads; head += warps) {
        const bool is_query = head < num_heads;
        const float* source = row_qkv + head * kAttentionHeadDim;
        const float* weight = is_query ? q_norm : k_norm;
        float values[kDimsPerLane];
        float squares = 0.0f;
#pragma unroll
        for (int i = 0; i < kDimsPerLane; ++i) {
            values[i] = source[lane + i * kWarpSize];
            squares += values[i] * values[i];
        }
        const float inverse_rms = 1.0f / sqrtf(warp_sum(squares) / kAttentionHeadDim + eps);
#pragma unroll
        for (int i = 0; i < kDimsPerLane; ++i) {
            const int d = lane + i * kWarpSize;
            values[i] = weight[d] * (values[i] * inverse_rms);
        }
        // values[0]（次元 l）と values[2]（l+64）、values[1]（l+32）と values[3]（l+96）が組。
        float rotated[kDimsPerLane];
#pragma unroll
        for (int i = 0; i < kDimsPerLane / 2; ++i) {
            const int d = lane + i * kWarpSize;  // 前半の次元
            const float angle = static_cast<float>(position) * inverse_frequency[d];
            const float cosine = cosf(angle);
            const float sine = sinf(angle);
            rotated[i] = values[i] * cosine - values[i + 2] * sine;
            rotated[i + 2] = values[i + 2] * cosine + values[i] * sine;
        }
        if (is_query) {
            float* destination = query + static_cast<size_t>(row) * num_heads * kAttentionHeadDim + head * kAttentionHeadDim;
#pragma unroll
            for (int i = 0; i < kDimsPerLane; ++i) destination[lane + i * kWarpSize] = rotated[i];
        } else {
            const size_t slot = static_cast<size_t>(position) * kv_width + (head - num_heads) * kAttentionHeadDim;
#pragma unroll
            for (int i = 0; i < kDimsPerLane; ++i) key_cache[slot + lane + i * kWarpSize] = __float2half_rn(rotated[i]);
        }
    }
    // v: FP16 にしてキャッシュへ。
    const float* value_source = row_qkv + num_heads * kAttentionHeadDim + kv_width;
    for (int i = threadIdx.x; i < kv_width; i += blockDim.x) {
        value_cache[static_cast<size_t>(position) * kv_width + i] = __float2half_rn(value_source[i]);
    }
    // ゲート: 1 ワープが 1 ヘッドずつ、hidden 個の内積。
    const float* row_normed = normed + static_cast<size_t>(row) * hidden_size;
    for (int head = warp; head < num_heads; head += warps) {
        const __half* weight = gate_weight + static_cast<size_t>(head) * hidden_size;
        float dot = 0.0f;
        for (int i = lane; i < hidden_size; i += kWarpSize) dot += __half2float(weight[i]) * row_normed[i];
        dot = warp_sum(dot);
        if (lane == 0) gate[static_cast<size_t>(row) * num_heads + head] = sigmoid(dot);
    }
}

// attention_prefill は、区切りのトークン（位置 start_position から tokens 個）の因果 attention を計算し、
// ゲートを掛けた出力（[トークン][num_heads × 128]、FP32）を書きます。KV キャッシュには attention_prepare_rows が
// 区切りの分まで書き終えている必要があります。
//
// 分担: スレッドブロック (g, b) が KV ヘッド g と、トークン 8b〜8b+7 を受け持ちます。ワープ w がトークン 8b+w の
// 4 ヘッド（g を共有する query ヘッド）を受け持ちます。見る位置の範囲を 32 位置ずつ K・V を共有メモリに読み、
//   1. score: レーン j が位置 j の 4 ヘッド分の内積を計算する（query は共有メモリから全レーンが同じ値を読む）
//   2. online softmax: ワープ内で最大値と合計を更新する
//   3. 重み付き和: レーン l が次元 4l〜4l+3 を受け持ち、各位置の重みをシャッフルで受け取って足す
// を繰り返します。見えない位置（未来の位置、窓の外）の score は -inf です。
// window が 0 なら全文脈、正なら「位置の差 < window」だけを見ます（attention_decode と同じ）。
// 起動: grid = (num_kv_heads, ceil(tokens / 8))、block = 256。head_dim は 128、num_heads = 4 × num_kv_heads。
__global__ void __launch_bounds__(kAttentionThreads) attention_prefill(
    const float* __restrict__ query, const __half* __restrict__ key_cache, const __half* __restrict__ value_cache,
    const float* __restrict__ gate, int start_position, int tokens, int window, int num_kv_heads, float scale,
    float* __restrict__ out) {
    __shared__ __align__(16) __half key_tile[kPrefillChunk][kPrefillKeyStride];
    __shared__ __align__(16) __half value_tile[kPrefillChunk][kAttentionHeadDim];
    __shared__ float query_tile[kPrefillTokensPerBlock][kAttentionGroup][kAttentionHeadDim];
    constexpr int kDimsPerLane = kAttentionHeadDim / kWarpSize;  // 4

    const int kv_head = blockIdx.x;
    const int lane = threadIdx.x % kWarpSize;
    const int warp = threadIdx.x / kWarpSize;
    const int num_heads = kAttentionGroup * num_kv_heads;
    const int kv_width = num_kv_heads * kAttentionHeadDim;
    const int first_head = kv_head * kAttentionGroup;
    const int first_token = blockIdx.y * kPrefillTokensPerBlock;
    const int token = first_token + warp;
    const bool active = token < tokens;
    const int position = start_position + token;
    const int first_visible = window > 0 ? max(0, position - window + 1) : 0;

    // このワープのトークンの 4 ヘッド分の query を共有メモリに置く。
    for (int i = lane; i < kAttentionGroup * kAttentionHeadDim; i += kWarpSize) {
        const int h = i / kAttentionHeadDim;
        const int d = i % kAttentionHeadDim;
        query_tile[warp][h][d] =
            active ? query[static_cast<size_t>(token) * num_heads * kAttentionHeadDim + (first_head + h) * kAttentionHeadDim + d]
                   : 0.0f;
    }

    // ブロック全体で見る位置の範囲 [range_begin, range_end)。最初のトークンの窓の始まりから、最後のトークンの位置まで。
    const int last_token = min(first_token + kPrefillTokensPerBlock, tokens) - 1;
    const int block_first_position = start_position + first_token;
    const int range_begin = window > 0 ? max(0, block_first_position - window + 1) : 0;
    const int range_end = start_position + last_token + 1;

    float running_max[kAttentionGroup];
    float running_total[kAttentionGroup];
    float acc[kAttentionGroup][kDimsPerLane];
#pragma unroll
    for (int h = 0; h < kAttentionGroup; ++h) {
        running_max[h] = -INFINITY;
        running_total[h] = 0.0f;
#pragma unroll
        for (int i = 0; i < kDimsPerLane; ++i) acc[h][i] = 0.0f;
    }

    for (int chunk = range_begin; chunk < range_end; chunk += kPrefillChunk) {
        const int count = min(kPrefillChunk, range_end - chunk);
        __syncthreads();  // 前の区切りの K・V を全員が使い終わる（最初は query_tile の書き込み）のを待つ
        // K・V を 32 位置 × 128 次元、FP16 のまま読む（1 スレッド 8 個 × 2 回ずつ）。
        for (int i = threadIdx.x; i < kPrefillChunk * (kAttentionHeadDim / 8); i += blockDim.x) {
            const int slot = i / (kAttentionHeadDim / 8);
            const int column = (i % (kAttentionHeadDim / 8)) * 8;
            uint4 key = make_uint4(0, 0, 0, 0);
            uint4 value = make_uint4(0, 0, 0, 0);
            if (slot < count) {
                const size_t offset = static_cast<size_t>(chunk + slot) * kv_width + kv_head * kAttentionHeadDim + column;
                key = __ldg(reinterpret_cast<const uint4*>(key_cache + offset));
                value = __ldg(reinterpret_cast<const uint4*>(value_cache + offset));
            }
            // K の行は詰め物のため 16 バイト境界にそろわないので、4 バイトずつ書く。
            const unsigned* key_words = reinterpret_cast<const unsigned*>(&key);
#pragma unroll
            for (int w = 0; w < 4; ++w) {
                *reinterpret_cast<unsigned*>(&key_tile[slot][column + 2 * w]) = key_words[w];
            }
            *reinterpret_cast<uint4*>(&value_tile[slot][column]) = value;
        }
        __syncthreads();

        // 段 1: レーン j が位置 chunk + j の score を 4 ヘッド分計算する。
        const int key_position = chunk + lane;
        const bool visible = active && lane < count && key_position <= position && key_position >= first_visible;
        float score[kAttentionGroup] = {0.0f, 0.0f, 0.0f, 0.0f};
#pragma unroll 8
        for (int d = 0; d < kAttentionHeadDim; d += 2) {
            const float2 key = __half22float2(*reinterpret_cast<const __half2*>(&key_tile[lane][d]));
#pragma unroll
            for (int h = 0; h < kAttentionGroup; ++h) {
                score[h] += query_tile[warp][h][d] * key.x + query_tile[warp][h][d + 1] * key.y;
            }
        }
        // 段 2: online softmax。見えない位置は重み 0。
        float weight[kAttentionGroup];
#pragma unroll
        for (int h = 0; h < kAttentionGroup; ++h) {
            const float value = visible ? score[h] * scale : -INFINITY;
            const float new_max = fmaxf(running_max[h], warp_max(value));
            // この区切りまで見える位置がなければ（new_max = -inf）、何も足さない。
            const float correction = new_max == -INFINITY ? 1.0f : expf(running_max[h] - new_max);
            weight[h] = visible ? expf(value - new_max) : 0.0f;
            running_total[h] = running_total[h] * correction + warp_sum(weight[h]);
            running_max[h] = new_max;
#pragma unroll
            for (int i = 0; i < kDimsPerLane; ++i) acc[h][i] *= correction;
        }
        // 段 3: 重み付き和。レーン l が次元 4l〜4l+3。
        for (int j = 0; j < count; ++j) {
            const uint2 packed = *reinterpret_cast<const uint2*>(&value_tile[j][lane * kDimsPerLane]);
            const float2 v01 = __half22float2(*reinterpret_cast<const __half2*>(&packed.x));
            const float2 v23 = __half22float2(*reinterpret_cast<const __half2*>(&packed.y));
#pragma unroll
            for (int h = 0; h < kAttentionGroup; ++h) {
                const float w = __shfl_sync(kFullMask, weight[h], j);
                acc[h][0] += w * v01.x;
                acc[h][1] += w * v01.y;
                acc[h][2] += w * v23.x;
                acc[h][3] += w * v23.y;
            }
        }
    }

    if (!active) return;
    float* row_out = out + static_cast<size_t>(token) * num_heads * kAttentionHeadDim;
#pragma unroll
    for (int h = 0; h < kAttentionGroup; ++h) {
        const float factor = gate[static_cast<size_t>(token) * num_heads + first_head + h] / running_total[h];
        float4 result = make_float4(acc[h][0] * factor, acc[h][1] * factor, acc[h][2] * factor, acc[h][3] * factor);
        *reinterpret_cast<float4*>(row_out + (first_head + h) * kAttentionHeadDim + lane * kDimsPerLane) = result;
    }
}

}  // extern "C"
