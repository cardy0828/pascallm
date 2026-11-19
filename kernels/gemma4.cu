// gemma4.cu
//
// Gemma 4（12B など）の生成（decode）と入力処理（prefill）に使うカーネルです（M7）。Go 側（internal/gemma4gpu）から
// 名前で呼びます。計算の手順は CPU の参照実装（internal/gemma4）と同じです。
//
// 行列とベクトルの積（gemv_pq4）、複数トークンの行列積（gemm_pq4）、argmax、選び方の候補の 2 段目（candidates_final）、
// 行の int8 化（quantize_rows）と、共通の道具（warp_sum・quantize_warp など）は lfm3.cu のものをそのまま使います。
// そのため lfm3.cu を取り込み、この cubin には LFM3 のカーネルも入ります（使わないだけで害はない）。
// 共通部分をヘッダーに分けるのは、後の整理とします。
//
// 共通の約束は lfm3.cu と同じです（状態 state は int32 × 4 で [0] 入力トークン、[1] 位置。活性値は 32 要素ごとに int8 化）。
//
// KV キャッシュは既定 FP16（比較用 cubin は K FP32 / V FP16 または K/V FP32）で、[スロット][kv_heads × head_dim] の並びです。位置 p のスロットは p mod ring_size です。
//   - 全文脈の層: ring_size は最大の位置数（輪にならない）
//   - 局所の層: ring_size は「窓 + 余裕」（古い位置を上書きする輪状バッファ）。余裕は、入力処理が 1 区切りを
//     まとめて書いても、区切りの中のトークンが見る位置を上書きしない大きさにします（Go 側で確かめる）

#include "lfm3.cu"

namespace {

// KV の格納型は cubin ごとに固定し、内側ループで実行時分岐を行わない。
#ifdef G4_KEY_FP32
using G4Key = float;
#else
using G4Key = __half;
#endif
#ifdef G4_VALUE_FP32
using G4Value = float;
#else
using G4Value = __half;
#endif
// g4_store_key は正規化・RoPE 後の K を選択した精度で保存する。FP32 は丸めない。
__device__ __forceinline__ G4Key g4_store_key(float value) {
#ifdef G4_KEY_FP32
    return value;
#else
    return __float2half_rn(value);
#endif
}
// g4_store_value は正規化後の V を格納する。FP32 は品質比較専用。
__device__ __forceinline__ G4Value g4_store_value(float value) {
#ifdef G4_VALUE_FP32
    return value;
#else
    return __float2half_rn(value);
#endif
}
// g4_load_value は格納型に応じて V を FP32 の積和へ渡す。
__device__ __forceinline__ float g4_load_value(G4Value value) {
#ifdef G4_VALUE_FP32
    return value;
#else
    return __half2float(value);
#endif
}


constexpr int kG4Threads = 256;                 // attention のカーネルの 1 スレッドブロックのスレッド数
constexpr int kG4Warps = kG4Threads / kWarpSize;  // 8
constexpr int kG4MaxDimsPerLane = 16;           // g4_prepare_vector で 1 レーンが持つ最大の次元数（head_dim ≤ 512）
constexpr int kG4MinPositionsPerSplit = 128;    // 位置の範囲を分割するとき、1 分割に入れる最小の位置数
constexpr int kG4PartialHeader = 2;             // 部分結果の先頭の float の数（最大値、合計）
constexpr int kG4KeyPadding = 2;                // 共有メモリの K の 1 行に足す FP16 の数（バンクの衝突を避ける）
constexpr int kG4HalvesPerLoad = 8;             // uint4 の 1 回の読み出しに入る FP16 の数
constexpr float kG4GeluCoefficient = 0.044715f;
constexpr float kG4SqrtTwoOverPi = 0.7978845608028654f;
constexpr int kG4CandidatePartThreads = 1024;   // g4_candidates_partial の 1 スレッドブロックのスレッド数
constexpr int kG4CandidatePartWarps = kG4CandidatePartThreads / kWarpSize;  // 32

// G4Range は、attention が見る位置の範囲と、その分け方です。
struct G4Range {
    int first;      // 見える最も古い位置
    int length;     // 見える位置の数
    int splits;     // 使う分割数（1 以上、max_splits 以下）
    int per_split;  // 1 分割の位置数
};

// g4_visible_range は、位置 position の query が見る範囲と分け方を返します。
// window が 0 なら全文脈、正なら「position − 位置 < window」の位置だけです（llama.cpp の is_masked_swa と同じ）。
// 分割数は、1 分割が kG4MinPositionsPerSplit 位置以上になるように決めます。
// g4_attention_* と g4_attention_combine が同じ結果を得るために、この関数を共有します。
// 例: position 1500、window 1024、max_splits 8 -> first 477、length 1024、splits 8、per_split 128
__device__ __forceinline__ G4Range g4_visible_range(int position, int window, int max_splits) {
    G4Range range;
    range.first = window > 0 ? max(0, position - window + 1) : 0;
    range.length = position - range.first + 1;
    range.splits = min(max_splits, (range.length + kG4MinPositionsPerSplit - 1) / kG4MinPositionsPerSplit);
    range.per_split = (range.length + range.splits - 1) / range.splits;
    return range;
}

// g4_embedding_value は、PQ4 の埋め込み表から、トークン token の次元 index の値（係数を掛けた復元値）を返します。
// バイト j の下位 4bit が要素 j、上位 4bit が要素 j + 16 です。復元値は (q − 8) × 係数。
// 例: hidden_size 3840 なら 1 行 120 ブロック。index 37 はブロック 1 の要素 5
__device__ __forceinline__ float g4_embedding_value(const uint8_t* __restrict__ packed,
                                                    const __half* __restrict__ scales, int token, int hidden_size,
                                                    int index) {
    constexpr int kBlockBytes = kQuantBlock / 2;  // 16
    const int blocks_per_row = hidden_size / kQuantBlock;
    const size_t block = static_cast<size_t>(token) * blocks_per_row + index / kQuantBlock;
    const int element = index % kQuantBlock;
    const uint8_t byte = packed[block * kBlockBytes + element % kBlockBytes];
    const int nibble = element < kBlockBytes ? (byte & 0x0F) : (byte >> 4);
    return (static_cast<float>(nibble) - kWeightCenter) * __half2float(scales[block]);
}

// g4_gelu は GELU の tanh による近似です（llama.cpp と同じ式）。例: g4_gelu(0) = 0、g4_gelu(1) ≒ 0.8412
__device__ __forceinline__ float g4_gelu(float x) {
    return 0.5f * x * (1.0f + tanhf(kG4SqrtTwoOverPi * x * (1.0f + kG4GeluCoefficient * x * x)));
}

// g4_residual_norm_quantize_vector は、残差の加算・層の出力の係数・次の正規化・int8 化をまとめて行います。
//
//   delta が null でなければ: hidden = (hidden + RMSNorm(delta) × delta_weight) × hidden_scale
//   次に: RMSNorm(hidden) × next_weight を 32 要素ごとに int8 化して q・info に書く
//
// hidden_scale は、FFN の後では層の出力の係数（layer_output_scale）、attention の後では 1 です。
// 1 スレッドブロック（1024 スレッド）でベクトル全体を処理します。n は 32 の倍数で、4096 以下。
// 作りは lfm3.cu の residual_norm_quantize_vector と同じです（必要な値を最初にまとめて読み、レジスタに置く）。
__device__ void g4_residual_norm_quantize_vector(float* __restrict__ hidden, const float* __restrict__ delta,
                                                 const float* __restrict__ delta_weight,
                                                 const float* __restrict__ next_weight, float eps, int n,
                                                 float hidden_scale, int8_t* __restrict__ q,
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
                residual[k] = (residual[k] + update_weight[k] * (update[k] * inverse_rms)) * hidden_scale;
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
            quantize_warp(output_weight[k] * (residual[k] * inverse_rms), i / kQuantBlock, q, info);
        }
    }
}

// g4_prepare_vector は、attention の前処理を、1 ワープが 1 本のベクトル（1 ヘッド分の q・k・v のどれか）について行います。
//
//   q: RMSNorm(q) × q_norm に RoPE を掛け、query_out（FP32）に書く
//   k: RMSNorm(k) × k_norm に RoPE を掛け、FP16 で KV キャッシュの位置 position のスロットに書く
//   v: RMSNorm(v)（重みなし、RoPE なし）を FP16 で KV キャッシュに書く
//
// ベクトルの番号 vector は、0〜num_heads−1 が q、続く kv_heads 個が k、その次の kv_heads 個が v です。
// qkv は 1 トークン分の GEMV の出力 [q（num_heads × head_dim）| k（kv_heads × head_dim）| …] で、v の元は
// qkv + value_offset から読みます（全文脈の層は v の行列がなく、value_offset が k と同じ位置を指す）。
// レーン l は次元 l, l+32, … を持ちます。RoPE で組になる次元 d と d + head_dim/2 が同じレーンにあるため、
// シャッフルが要りません。角度は「位置 × 逆周波数」（float32）です。
// ワープの全レーンで呼びます。head_dim は 64 の倍数で 512 以下。
// 例: 局所の層（head_dim 256、num_heads 16、kv_heads 8）なら vector 0〜15 が q、16〜23 が k、24〜31 が v
__device__ void g4_prepare_vector(const float* __restrict__ qkv, const float* __restrict__ q_norm,
                                  const float* __restrict__ k_norm, const float* __restrict__ inverse_frequency,
                                  float eps, int position, int head_dim, int num_heads, int kv_heads,
                                  int value_offset, float* __restrict__ query_out, G4Key* __restrict__ key_cache,
                                  G4Value* __restrict__ value_cache, int ring_size, int vector) {
    const int lane = threadIdx.x % kWarpSize;
    const int dims_per_lane = head_dim / kWarpSize;
    const int pairs_per_lane = dims_per_lane / 2;
    const bool is_query = vector < num_heads;
    const bool is_key = !is_query && vector < num_heads + kv_heads;
    const int kv_head = is_query ? 0 : (is_key ? vector - num_heads : vector - num_heads - kv_heads);
    const float* source = qkv + vector * head_dim;  // q と k は、この並びのまま
    if (!is_query && !is_key) source = qkv + value_offset + kv_head * head_dim;

    float values[kG4MaxDimsPerLane];
    float squares = 0.0f;
#pragma unroll
    for (int i = 0; i < kG4MaxDimsPerLane; ++i) {
        values[i] = i < dims_per_lane ? source[lane + i * kWarpSize] : 0.0f;
        squares += values[i] * values[i];
    }
    const float inverse_rms = 1.0f / sqrtf(warp_sum(squares) / head_dim + eps);
    const size_t slot = static_cast<size_t>(position % ring_size) * (kv_heads * head_dim) + kv_head * head_dim;

    if (!is_query && !is_key) {
#pragma unroll
        for (int i = 0; i < kG4MaxDimsPerLane; ++i) {
            if (i < dims_per_lane) value_cache[slot + lane + i * kWarpSize] = g4_store_value(values[i] * inverse_rms);
        }
        return;
    }
    const float* weight = is_query ? q_norm : k_norm;
#pragma unroll
    for (int i = 0; i < kG4MaxDimsPerLane; ++i) {
        if (i < dims_per_lane) values[i] = weight[lane + i * kWarpSize] * (values[i] * inverse_rms);
    }
    // values[i]（次元 l + 32i）と values[i + pairs_per_lane]（次元 l + 32i + head_dim/2）が組。
#pragma unroll
    for (int i = 0; i < kG4MaxDimsPerLane / 2; ++i) {
        if (i < pairs_per_lane) {
            const float angle = static_cast<float>(position) * inverse_frequency[lane + i * kWarpSize];
            const float cosine = cosf(angle);
            const float sine = sinf(angle);
            const float first = values[i];
            const float second = values[i + pairs_per_lane];
            values[i] = first * cosine - second * sine;
            values[i + pairs_per_lane] = second * cosine + first * sine;
        }
    }
#pragma unroll
    for (int i = 0; i < kG4MaxDimsPerLane; ++i) {
        if (i >= dims_per_lane) continue;
        const int d = lane + i * kWarpSize;
        if (is_query) {
            query_out[vector * head_dim + d] = values[i];
        } else {
            key_cache[slot + d] = g4_store_key(values[i]);
        }
    }
}

// プリフィルで使う KV スロットの求め方。通常経路は任意長のリングに対応する。
enum class G4Slot { Ring, PowerOfTwoRing, Linear };

// g4_slot はトークン位置を KV スロットに変換する。線形領域の位置は呼び出し側の上限検査済み。
template <G4Slot mode>
__device__ __forceinline__ int g4_slot(int position, int ring_size) {
    if (mode == G4Slot::Linear) return position;
    if (mode == G4Slot::PowerOfTwoRing) return position & (ring_size - 1);
    return position % ring_size;
}

// g4_attend_range は、kGroup 個の query ヘッド（同じ KV ヘッドを使う）について、位置 [begin, end) に対する
// attention の途中結果を求めます。スレッドブロック（256 スレッド）の全スレッドで呼びます。
//
//   acc[h][k]      スレッドが受け持つ次元（thread × kDimsPerThread + k）の Σ 重み × V（最大値を基準にした重み）
//   head_max[h]    スコアの最大値、head_total[h] 重みの合計（共有メモリ。呼び出し側が kGroup 個を用意する）
//
// 正規化した出力は acc / head_total です。範囲を分けたときは、最大値をそろえてから足します（g4_attention_combine）。
//
// 範囲の中は kPositions 位置ずつ、次の 3 段で処理し、最大値・合計・重み付き和を逐次更新します（online softmax）。
//   1. K と V を kPositions 位置分、FP16 のまま共有メモリに読む
//   2. スコア: スレッドが 1 位置の kSegmentDims 次元分（kGroup ヘッド）の内積を計算し、ワープ h がヘッド h の分を足して重みにする
//   3. 重み付き和: スレッドが自分の次元について、各位置の V に重みを掛けて足す
// 256 スレッドは「位置 kPositions 個 × 次元の区分 kSegments 個」に分かれ、1 区分は 16 次元です
// （局所: 16 位置 × 16 区分、全文脈: 8 位置 × 32 区分。共有メモリを小さくして同時実行数を増やす）。
// query は [kGroup][kHeadDim] の FP32、kv_offset は KV キャッシュの 1 スロットの中での KV ヘッドの位置です。
template <int kHeadDim, int kGroup, int kPositions, G4Slot slotMode = G4Slot::Ring>
__device__ void g4_attend_range(const float* __restrict__ query, const G4Key* __restrict__ key_cache,
                                const G4Value* __restrict__ value_cache, int kv_offset, int kv_width, int ring_size,
                                int begin, int end, float scale, float (&acc)[kGroup][kHeadDim / kG4Threads],
                                float* head_max, float* head_total) {
    constexpr int kSegments = kG4Threads / kPositions;
    constexpr int kSegmentDims = kHeadDim / kSegments;
    constexpr int kDimsPerThread = kHeadDim / kG4Threads;
    constexpr int kLoadsPerRow = kHeadDim / kG4HalvesPerLoad;  // 1 位置を読む uint4 の数
    // 全文脈プリフィルの K は uint4 でまとめて共有メモリに置く。FP16 の 8 要素で 16 バイト境界を保つ。
    constexpr int kKeyStride = kHeadDim + (slotMode == G4Slot::Linear ? kG4HalvesPerLoad : kG4KeyPadding);
    static_assert(kSegmentDims % 2 == 0, "half2 単位で区分を処理する");
    static_assert(kGroup <= kG4Warps && kPositions <= kWarpSize, "ワープ h がヘッド h、レーンが位置を受け持つ");
    __shared__ float query_tile[kGroup][kHeadDim];
    __shared__ __align__(16) G4Key key_tile[kPositions][kKeyStride];
    __shared__ __align__(16) G4Value value_tile[kPositions][kHeadDim];
    __shared__ float score_parts[kGroup][kG4Threads];
    __shared__ float weights[kGroup][kPositions];
    __shared__ float corrections[kGroup];

    const int thread = threadIdx.x;
    const int lane = thread % kWarpSize;
    const int warp = thread / kWarpSize;
    const int score_position = thread % kPositions;
    const int segment = thread / kPositions;

    for (int i = thread; i < kGroup * kHeadDim; i += kG4Threads) query_tile[i / kHeadDim][i % kHeadDim] = query[i];
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
#pragma unroll
        for (int k = 0; k < kDimsPerThread; ++k) acc[h][k] = 0.0f;
    }
    // ワープ h < kGroup がヘッド h について持つ状態（ワープ内の全レーンで同じ値）。
    float running_max = -INFINITY;
    float running_total = 0.0f;

    for (int chunk = begin; chunk < end; chunk += kPositions) {
        const int count = min(kPositions, end - chunk);
        __syncthreads();  // query_tile の書き込み、または前の区切りの K・V・重みを全員が使い終わるのを待つ
#if defined(G4_KEY_FP32) || defined(G4_VALUE_FP32)
        // 128 bit 単位で各格納型を読み、K の行の詰め物を考慮して共有メモリへ保存する。
        constexpr int key_elements_per_load = sizeof(uint4) / sizeof(G4Key);
        constexpr int value_elements_per_load = sizeof(uint4) / sizeof(G4Value);
        for (int i = thread; i < kPositions * kHeadDim / key_elements_per_load; i += kG4Threads) {
            const int slot = i / (kHeadDim / key_elements_per_load);
            const int column = (i % (kHeadDim / key_elements_per_load)) * key_elements_per_load;
            uint4 packed = make_uint4(0, 0, 0, 0);
            if (slot < count) {
                const size_t offset = static_cast<size_t>(g4_slot<slotMode>(chunk + slot, ring_size)) * kv_width + kv_offset + column;
                packed = __ldg(reinterpret_cast<const uint4*>(key_cache + offset));
            }
            const unsigned* words = reinterpret_cast<const unsigned*>(&packed);
#pragma unroll
            for (int w = 0; w < 4; ++w)
                reinterpret_cast<unsigned*>(&key_tile[slot][column])[w] = words[w];
        }
        for (int i = thread; i < kPositions * kHeadDim / value_elements_per_load; i += kG4Threads) {
            const int slot = i / (kHeadDim / value_elements_per_load);
            const int column = (i % (kHeadDim / value_elements_per_load)) * value_elements_per_load;
            uint4 packed = make_uint4(0, 0, 0, 0);
            if (slot < count) {
                const size_t offset = static_cast<size_t>(g4_slot<slotMode>(chunk + slot, ring_size)) * kv_width + kv_offset + column;
                packed = __ldg(reinterpret_cast<const uint4*>(value_cache + offset));
            }
            *reinterpret_cast<uint4*>(&value_tile[slot][column]) = packed;
        }
#else
        // 段 1: K・V を読む（範囲外は 0）。K の行は詰め物のため 16 バイト境界にそろわないので、4 バイトずつ書く。
        for (int i = thread; i < kPositions * kLoadsPerRow; i += kG4Threads) {
            const int slot = i / kLoadsPerRow;
            const int column = (i % kLoadsPerRow) * kG4HalvesPerLoad;
            uint4 key = make_uint4(0, 0, 0, 0);
            uint4 value = make_uint4(0, 0, 0, 0);
            if (slot < count) {
                const size_t offset =
                    static_cast<size_t>(g4_slot<slotMode>(chunk + slot, ring_size)) * kv_width + kv_offset + column;
                key = __ldg(reinterpret_cast<const uint4*>(key_cache + offset));
                value = __ldg(reinterpret_cast<const uint4*>(value_cache + offset));
            }
            if (slotMode == G4Slot::Linear) {
                *reinterpret_cast<uint4*>(&key_tile[slot][column]) = key;
            } else {
                const unsigned* key_words = reinterpret_cast<const unsigned*>(&key);
#pragma unroll
                for (int w = 0; w < 4; ++w) {
                    *reinterpret_cast<unsigned*>(&key_tile[slot][column + 2 * w]) = key_words[w];
                }
            }
            *reinterpret_cast<uint4*>(&value_tile[slot][column]) = value;
        }
#endif
        __syncthreads();

        // 段 2a: 位置 score_position の、次元 segment × kSegmentDims から kSegmentDims 個分の内積（kGroup ヘッド）。
        float dots[kGroup];
#pragma unroll
        for (int h = 0; h < kGroup; ++h) dots[h] = 0.0f;
        const G4Key* key_row = &key_tile[score_position][segment * kSegmentDims];
#pragma unroll 8
        for (int j = 0; j < kSegmentDims; j += 2) {
#ifdef G4_KEY_FP32
            const float2 key = make_float2(key_row[j], key_row[j + 1]);
#else
            const float2 key = __half22float2(*reinterpret_cast<const __half2*>(key_row + j));
#endif
#pragma unroll
            for (int h = 0; h < kGroup; ++h) {
                dots[h] += query_tile[h][segment * kSegmentDims + j] * key.x +
                           query_tile[h][segment * kSegmentDims + j + 1] * key.y;
            }
        }
#pragma unroll
        for (int h = 0; h < kGroup; ++h) score_parts[h][thread] = dots[h];
        __syncthreads();

        // 段 2b: ワープ h がヘッド h の重みを求める（レーンが位置）。最大値の更新に合わせて、これまでの合計と
        // 重み付き和を縮める係数 correction を、段 3 のために共有メモリに置く。範囲外の位置の重みは 0。
        if (warp < kGroup) {
            const bool active = lane < count;
            float score = 0.0f;
            if (active) {
#pragma unroll
                for (int s = 0; s < kSegments; ++s) score += score_parts[warp][s * kPositions + lane];
            }
            const float value = active ? score * scale : -INFINITY;
            const float new_max = fmaxf(running_max, warp_max(value));
            const float correction = expf(running_max - new_max);  // 最初は exp(-inf) = 0
            const float weight = active ? expf(value - new_max) : 0.0f;
            running_total = running_total * correction + warp_sum(weight);
            running_max = new_max;
            if (lane < kPositions) weights[warp][lane] = weight;
            if (lane == 0) corrections[warp] = correction;
        }
        __syncthreads();

        // 段 3: 重み付き和。
#pragma unroll
        for (int h = 0; h < kGroup; ++h) {
            const float factor = corrections[h];
#pragma unroll
            for (int k = 0; k < kDimsPerThread; ++k) acc[h][k] *= factor;
        }
        // 全8位置が有効なタイルではループを展開する。末尾の端数は有効な位置だけを同じ順序で加算する。
        auto accumulate_position = [&](int p) {
#pragma unroll
            for (int k = 0; k < kDimsPerThread; ++k) {
                const float v = g4_load_value(value_tile[p][thread * kDimsPerThread + k]);
#pragma unroll
                for (int h = 0; h < kGroup; ++h) acc[h][k] += weights[h][p] * v;
            }
        };
        if (slotMode == G4Slot::Linear && count == kPositions) {
#pragma unroll
            for (int p = 0; p < kPositions; ++p) accumulate_position(p);
        } else {
            for (int p = 0; p < count; ++p) accumulate_position(p);
        }
    }
    __syncthreads();
    if (warp < kGroup && lane == 0) {
        head_max[warp] = running_max;
        head_total[warp] = running_total;
    }
    __syncthreads();
}

// g4_attention_decode_body は、生成（1 トークン）用の attention 本体です。スレッドブロック (g, s) が、
// query ヘッド g × kGroup から kGroup 個と、見る位置の範囲を分けた s 番目を受け持ち、途中結果を partial に書きます。
// partial の並びは [組 g][分割 s][ヘッド h][最大値, 合計, 重み付き和 kHeadDim 個] です（分割は gridDim.y 個分を確保）。
// 使わない分割（範囲が短いとき）のブロックは何もしません。結果の合成と int8 化は g4_attention_combine が行います。
template <int kHeadDim, int kGroup, int kPositions>
__device__ void g4_attention_decode_body(const float* __restrict__ query, const G4Key* __restrict__ key_cache,
                                         const G4Value* __restrict__ value_cache, const int* __restrict__ state,
                                         int window, int ring_size, int kv_heads, int heads_per_kv, float scale,
                                         float* __restrict__ partial) {
    constexpr int kDimsPerThread = kHeadDim / kG4Threads;
    constexpr int kPartialStride = kG4PartialHeader + kHeadDim;
    __shared__ float head_max[kGroup];
    __shared__ float head_total[kGroup];
    const int group = blockIdx.x;
    const int split = blockIdx.y;
    const int position = state[1];
    const G4Range range = g4_visible_range(position, window, gridDim.y);
    if (split >= range.splits) return;  // ブロック全体で同じ条件なので、__syncthreads の前に抜けてよい
    const int begin = range.first + split * range.per_split;
    const int end = min(begin + range.per_split, position + 1);
    const int kv_head = group * kGroup / heads_per_kv;

    float acc[kGroup][kDimsPerThread];
    g4_attend_range<kHeadDim, kGroup, kPositions>(query + group * kGroup * kHeadDim, key_cache, value_cache,
                                                  kv_head * kHeadDim, kv_heads * kHeadDim, ring_size, begin, end, scale,
                                                  acc, head_max, head_total);
    float* mine = partial + (static_cast<size_t>(group) * gridDim.y + split) * kGroup * kPartialStride;
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
#pragma unroll
        for (int k = 0; k < kDimsPerThread; ++k) {
            mine[h * kPartialStride + kG4PartialHeader + threadIdx.x * kDimsPerThread + k] = acc[h][k];
        }
    }
    if (threadIdx.x < kGroup) {
        mine[threadIdx.x * kPartialStride] = head_max[threadIdx.x];
        mine[threadIdx.x * kPartialStride + 1] = head_total[threadIdx.x];
    }
}

// g4_attention_rows_body は、入力処理用の attention 本体です。スレッドブロック (g, t) が、区切りのトークン t
// （位置 start_position + t）の query ヘッド g × kGroup から kGroup 個を受け持ち、見える範囲を分けずに計算して、
// 正規化した出力を out（[トークン][num_heads × kHeadDim]、FP32）に書きます。
// KV キャッシュには g4_attention_prepare_rows が区切りの分まで書き終えている必要があります。
template <int kHeadDim, int kGroup, int kPositions, G4Slot slotMode = G4Slot::Ring>
__device__ void g4_attention_rows_body(const float* __restrict__ query, const G4Key* __restrict__ key_cache,
                                       const G4Value* __restrict__ value_cache, int start_position, int window,
                                       int ring_size, int kv_heads, int heads_per_kv, int num_heads, float scale,
                                       float* __restrict__ out) {
    constexpr int kDimsPerThread = kHeadDim / kG4Threads;
    __shared__ float head_max[kGroup];
    __shared__ float head_total[kGroup];
    // 全文脈の専用カーネルは隣接行を grid.x に並べ、同じ K/V をキャッシュから読みやすくする。
    const int group = slotMode == G4Slot::Linear ? blockIdx.y : blockIdx.x;
    const size_t row = slotMode == G4Slot::Linear ? blockIdx.x : blockIdx.y;
    const int position = start_position + static_cast<int>(row);
    const int first = window > 0 ? max(0, position - window + 1) : 0;
    const int kv_head = group * kGroup / heads_per_kv;
    const size_t row_offset = row * num_heads * kHeadDim + group * kGroup * kHeadDim;

    float acc[kGroup][kDimsPerThread];
    g4_attend_range<kHeadDim, kGroup, kPositions, slotMode>(query + row_offset, key_cache, value_cache, kv_head * kHeadDim,
                                                  kv_heads * kHeadDim, ring_size, first, position + 1, scale, acc,
                                                  head_max, head_total);
#pragma unroll
    for (int h = 0; h < kGroup; ++h) {
        const float inverse_total = 1.0f / head_total[h];
#pragma unroll
        for (int k = 0; k < kDimsPerThread; ++k) {
            out[row_offset + h * kHeadDim + threadIdx.x * kDimsPerThread + k] = acc[h][k] * inverse_total;
        }
    }
}

}  // namespace

extern "C" {

// ===== 生成（1 トークン）=====

// g4_embed_pq4 は、state[0] のトークンの埋め込み（PQ4 の表を復元）に embedding_scale を掛けて hidden に書きます。
// 埋め込み表は語彙出力と同じ PQ4 の行列です。起動: grid = hidden / 256、block = 256。例: hidden 3840 なら grid = 15
__global__ void g4_embed_pq4(const uint8_t* __restrict__ packed, const __half* __restrict__ scales,
                             const int* __restrict__ state, float* __restrict__ hidden, int hidden_size,
                             float embedding_scale) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_size) return;
    hidden[index] = g4_embedding_value(packed, scales, state[0], hidden_size, index) * embedding_scale;
}

// g4_residual_norm_quantize は、生成用の残差・係数・正規化・int8 化です（本体は g4_residual_norm_quantize_vector）。
// 起動: grid = 1、block = 1024
__global__ void g4_residual_norm_quantize(float* __restrict__ hidden, const float* __restrict__ delta,
                                          const float* __restrict__ delta_weight,
                                          const float* __restrict__ next_weight, float eps, int n, float hidden_scale,
                                          int8_t* __restrict__ q, ActBlockInfo* __restrict__ info) {
    g4_residual_norm_quantize_vector(hidden, delta, delta_weight, next_weight, eps, n, hidden_scale, q, info);
}

// g4_attention_prepare は、生成用の attention の前処理です（本体は g4_prepare_vector）。位置は state[1] です。
// ブロック b のワープ w がベクトル 8b + w を受け持ちます。
// 起動: grid = ceil((num_heads + 2 × kv_heads) / 8)、block = 256。例: 局所の層は 32 本で grid = 4、全文脈は 18 本で 3
__global__ void g4_attention_prepare(const float* __restrict__ qkv, const float* __restrict__ q_norm,
                                     const float* __restrict__ k_norm, const float* __restrict__ inverse_frequency,
                                     float eps, const int* __restrict__ state, int head_dim, int num_heads,
                                     int kv_heads, int value_offset, float* __restrict__ query_out,
                                     G4Key* __restrict__ key_cache, G4Value* __restrict__ value_cache, int ring_size) {
    const int vector = blockIdx.x * kG4Warps + threadIdx.x / kWarpSize;
    if (vector >= num_heads + 2 * kv_heads) return;
    g4_prepare_vector(qkv, q_norm, k_norm, inverse_frequency, eps, state[1], head_dim, num_heads, kv_heads,
                      value_offset, query_out, key_cache, value_cache, ring_size, vector);
}

// g4_attention_local は、局所の層（head_dim 256、1 個の KV ヘッドを query 2 ヘッドが使う）の attention 本体です。
// 起動: grid = (num_heads / 2, 最大の分割数)、block = 256。query は g4_attention_prepare の出力。
// 例: 16 ヘッド、分割 8 なら grid = (8, 8)。heads_per_kv = num_heads / kv_heads = 2
__global__ void __launch_bounds__(kG4Threads) g4_attention_local(
    const float* __restrict__ query, const G4Key* __restrict__ key_cache, const G4Value* __restrict__ value_cache,
    const int* __restrict__ state, int window, int ring_size, int kv_heads, int heads_per_kv, float scale,
    float* __restrict__ partial) {
    g4_attention_decode_body<256, 2, 16>(query, key_cache, value_cache, state, window, ring_size, kv_heads,
                                         heads_per_kv, scale, partial);
}

// g4_attention_global は、全文脈の層（head_dim 512）の attention 本体です。1 ブロックが query 4 ヘッドを受け持ちます
// （16 ヘッドが 1 個の KV ヘッドを使うので、同じ K・V を 4 ブロックが読む）。
// 起動: grid = (num_heads / 4, 最大の分割数)、block = 256。例: 16 ヘッド、分割 16 なら grid = (4, 16)
__global__ void __launch_bounds__(kG4Threads) g4_attention_global(
    const float* __restrict__ query, const G4Key* __restrict__ key_cache, const G4Value* __restrict__ value_cache,
    const int* __restrict__ state, int window, int ring_size, int kv_heads, int heads_per_kv, float scale,
    float* __restrict__ partial) {
    g4_attention_decode_body<512, 4, 8>(query, key_cache, value_cache, state, window, ring_size, kv_heads,
                                         heads_per_kv, scale, partial);
}

// g4_attention_combine は、g4_attention_local・g4_attention_global が書いた途中結果を合成し、attention の出力
// （num_heads × head_dim）を int8 化して q・info に書きます（o の射影の入力になる）。
// 分割ごとの最大値を全体の最大値にそろえてから、合計と重み付き和を足して割ります。
// group_size は 1 ブロックが受け持った query ヘッドの数（局所 2、全文脈 4）、max_splits は attention 本体の grid の y です。
// 起動: grid = num_heads × head_dim / 256、block = 256（1 スレッド 1 要素、1 ワープが int8 化の 1 ブロック）
__device__ void g4_combine_body(const float* __restrict__ partial, const int* __restrict__ state, int window,
                                     int max_splits, int head_dim, int group_size, int num_heads,
                                     int8_t* __restrict__ q, ActBlockInfo* __restrict__ info) {
    const int element = blockIdx.x * blockDim.x + threadIdx.x;
    if (element >= num_heads * head_dim) return;  // 32 の倍数なので、ワープ単位で抜ける
    const int head = element / head_dim;
    const int d = element % head_dim;
    const int group = head / group_size;
    const int local_head = head % group_size;
    const int stride = kG4PartialHeader + head_dim;
    const int split_stride = group_size * stride;
    const G4Range range = g4_visible_range(state[1], window, max_splits);
    const float* base = partial + static_cast<size_t>(group) * max_splits * split_stride + local_head * stride;
    float overall_max = -INFINITY;
    for (int s = 0; s < range.splits; ++s) overall_max = fmaxf(overall_max, base[s * split_stride]);
    float total = 0.0f;
    float sum = 0.0f;
    for (int s = 0; s < range.splits; ++s) {
        const float* part = base + s * split_stride;
        const float factor = expf(part[0] - overall_max);
        total += part[1] * factor;
        sum += part[kG4PartialHeader + d] * factor;
    }
    quantize_warp(sum / total, element / kQuantBlock, q, info);
}

// g4_gelu_mul_quantize は、FFN の中間値 gelu(gate) × up を計算して int8 化します（down の入力）。
// gate_up は [gate（n 個）, up（n 個）] の順に並んだ GEMV の出力です。
// 起動: grid = n / 256、block = 256。例: n = 15360 なら grid = 60
__global__ void g4_gelu_mul_quantize(const float* __restrict__ gate_up, int n, int8_t* __restrict__ q,
                                     ActBlockInfo* __restrict__ info) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;  // n は 32 の倍数なので、ワープ単位で抜ける
    quantize_warp(g4_gelu(gate_up[index]) * gate_up[n + index], index / kQuantBlock, q, info);
}

// g4_softcap は logits[i] = limit × tanh(logits[i] / limit) を計算します（値の大小の順は変わらない）。
// 起動: grid = ceil(n / 256)、block = 256。例: n = 262144、limit = 30
__global__ void g4_softcap(float* __restrict__ logits, int n, float limit) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    logits[index] = limit * tanhf(logits[index] / limit);
}

// g4_candidates_partial は、選び方の候補を絞る 1 段目です。lfm3.cu の candidates_partial と同じ処理で、
// 1 ブロックを 1024 スレッド（32 ワープ）にして、1 区間に 4096 個まで入るようにしたものです
// （語彙 262,144 個を 64 区間で扱うため）。2 段目は candidates_final をそのまま使います。
// 起動: grid = 64、block = kG4CandidatePartThreads（1024）。valid_tokens ≤ 64 × 4096。
__global__ void g4_candidates_partial(const float* __restrict__ logits, int valid_tokens,
                                      unsigned char* __restrict__ seen, int* __restrict__ counts, int accept_token,
                                      float inverse_temperature, float repetition_penalty, float presence_penalty,
                                      float frequency_penalty, float* __restrict__ part_values,
                                      int* __restrict__ part_ids, float* __restrict__ part_max,
                                      float* __restrict__ part_sum) {
    __shared__ Candidate lists[kG4CandidatePartWarps / 2 * kCandidateCount];
    __shared__ float scratch[kWarpSize];
    const int span = (valid_tokens + gridDim.x - 1) / gridDim.x;
    const int begin = blockIdx.x * span;
    const int end = min(begin + span, valid_tokens);
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
    const float block_total = block_reduce_sum(local_sum, scratch);

    warp_sort_candidates(items);
    merge_warp_candidates(items, lists, kG4CandidatePartWarps);
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
        part_sum[blockIdx.x] = block_total;
    }
}

// ===== 入力処理（prefill）=====
// 行のデータはトークンごとに連続して並びます（lfm3.cu の入力処理と同じ）。

// g4_embed_rows は、tokens[t] の埋め込みに embedding_scale を掛けて hidden の行 t に書きます。
// 起動: grid = (hidden / 256, トークン数)、block = 256
__global__ void g4_embed_rows(const uint8_t* __restrict__ packed, const __half* __restrict__ scales,
                              const int* __restrict__ tokens, float* __restrict__ hidden, int hidden_size,
                              float embedding_scale) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= hidden_size) return;
    const size_t row = blockIdx.y;
    hidden[row * hidden_size + index] =
        g4_embedding_value(packed, scales, tokens[row], hidden_size, index) * embedding_scale;
}

// g4_residual_norm_quantize_rows は、g4_residual_norm_quantize をトークンの行ごとに行います（ブロック t が行 t）。
// 起動: grid = トークン数、block = 1024
__global__ void g4_residual_norm_quantize_rows(float* __restrict__ hidden, const float* __restrict__ delta,
                                               const float* __restrict__ delta_weight,
                                               const float* __restrict__ next_weight, float eps, int n,
                                               float hidden_scale, int8_t* __restrict__ q,
                                               ActBlockInfo* __restrict__ info) {
    const size_t row = blockIdx.x;
    g4_residual_norm_quantize_vector(hidden + row * n, delta != nullptr ? delta + row * n : nullptr, delta_weight,
                                     next_weight, eps, n, hidden_scale, q + row * n, info + row * (n / kQuantBlock));
}

// g4_gelu_mul_quantize_rows は、g4_gelu_mul_quantize をトークンの行ごとに行います。gate_up は [トークン][2 × n] です。
// 起動: grid = (n / 256, トークン数)、block = 256
__global__ void g4_gelu_mul_quantize_rows(const float* __restrict__ gate_up, int n, int8_t* __restrict__ q,
                                          ActBlockInfo* __restrict__ info) {
    const int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= n) return;
    const size_t row = blockIdx.y;
    const float* row_values = gate_up + row * 2 * n;
    quantize_warp(g4_gelu(row_values[index]) * row_values[n + index], index / kQuantBlock, q + row * n,
                  info + row * (n / kQuantBlock));
}

// g4_attention_prepare_rows は、attention の前処理をトークンの行ごとに行います（位置は start_position + 行）。
// qkv は [トークン][qkv_width]、query_out は [トークン][num_heads × head_dim] です。
// 起動: grid = (ceil((num_heads + 2 × kv_heads) / 8), トークン数)、block = 256
__global__ void g4_attention_prepare_rows(const float* __restrict__ qkv, const float* __restrict__ q_norm,
                                          const float* __restrict__ k_norm,
                                          const float* __restrict__ inverse_frequency, float eps, int start_position,
                                          int head_dim, int num_heads, int kv_heads, int value_offset, int qkv_width,
                                          float* __restrict__ query_out, G4Key* __restrict__ key_cache,
                                          G4Value* __restrict__ value_cache, int ring_size) {
    const int vector = blockIdx.x * kG4Warps + threadIdx.x / kWarpSize;
    if (vector >= num_heads + 2 * kv_heads) return;
    const size_t row = blockIdx.y;
    g4_prepare_vector(qkv + row * qkv_width, q_norm, k_norm, inverse_frequency, eps,
                      start_position + static_cast<int>(row), head_dim, num_heads, kv_heads, value_offset,
                      query_out + row * num_heads * head_dim, key_cache, value_cache, ring_size, vector);
}

// g4_attention_local_rows は、局所の層の入力処理用の attention です（本体は g4_attention_rows_body）。
// 起動: grid = (num_heads / 2, トークン数)、block = 256
__global__ void __launch_bounds__(kG4Threads) g4_attention_local_rows(
    const float* __restrict__ query, const G4Key* __restrict__ key_cache, const G4Value* __restrict__ value_cache,
    int start_position, int window, int ring_size, int kv_heads, int heads_per_kv, int num_heads, float scale,
    float* __restrict__ out) {
    g4_attention_rows_body<256, 2, 16>(query, key_cache, value_cache, start_position, window, ring_size, kv_heads,
                                       heads_per_kv, num_heads, scale, out);
}

// g4_attention_global_rows は、全文脈の層の入力処理用の attention です。
// 起動: grid = (num_heads / 4, トークン数)、block = 256
__global__ void __launch_bounds__(kG4Threads) g4_attention_global_rows(
    const float* __restrict__ query, const G4Key* __restrict__ key_cache, const G4Value* __restrict__ value_cache,
    int start_position, int window, int ring_size, int kv_heads, int heads_per_kv, int num_heads, float scale,
    float* __restrict__ out) {
    g4_attention_rows_body<512, 4, 8>(query, key_cache, value_cache, start_position, window, ring_size, kv_heads,
                                       heads_per_kv, num_heads, scale, out);
}

#if !defined(G4_KEY_FP32) && !defined(G4_VALUE_FP32)
// 入力の局所 attention。例えば 2048 スロットのリングをビットマスクで参照する。
// 起動側はスロット数が 2 の累乗のときだけこのカーネルを選ぶ。
__global__ void __launch_bounds__(kG4Threads) g4_prefill_local_ring(
    const float* __restrict__ query, const G4Key* __restrict__ key_cache, const G4Value* __restrict__ value_cache,
    int start_position, int window, int ring_size, int kv_heads, int heads_per_kv, int num_heads, float scale,
    float* __restrict__ out) {
    g4_attention_rows_body<256, 2, 16, G4Slot::PowerOfTwoRing>(
        query, key_cache, value_cache, start_position, window, ring_size, kv_heads, heads_per_kv, num_heads, scale, out);
}

// 入力の全文脈 attention。8 query ヘッドで 1 回の K/V 読み込みを共有する。
// grid は (入力行数, ヘッド数 / 8)。各行から見える履歴は常に ring_size 内に収まる。
__global__ void __launch_bounds__(kG4Threads) g4_prefill_global8(
    const float* __restrict__ query, const G4Key* __restrict__ key_cache, const G4Value* __restrict__ value_cache,
    int start_position, int window, int ring_size, int kv_heads, int heads_per_kv, int num_heads, float scale,
    float* __restrict__ out) {
    g4_attention_rows_body<512, 8, 8, G4Slot::Linear>(
        query, key_cache, value_cache, start_position, window, ring_size, kv_heads, heads_per_kv, num_heads, scale, out);
}
#endif

}  // extern "C"

// g4_gemm_small は少数トークンの行列積です。1 warp が 4 行 x 最大 4 トークンを担当し、
// 同じ重みを 4 トークンで再利用します。各出力の加算順は gemv_pq4 と同じです。
// 起動: grid=(ceil(rows/16),ceil(tokens/4)), block=(32,4)。例: tokens=3 は 1 タイルです。
template<int tokens_per_tile>
__device__ __forceinline__ void g4_gemm_small_body(const int4* __restrict__ packed, const __half* __restrict__ scales,
    const int8_t* __restrict__ act_q, const ActBlockInfo* __restrict__ act_info,
    float* __restrict__ out, int rows, int blocks_per_row, int tokens, int input_block_shift) {
    constexpr int rows_per_warp = 4;
    const int first_row = (blockIdx.x * 4 + threadIdx.y) * rows_per_warp;
    const int first_token = blockIdx.y * tokens_per_tile;
    const int count = min(tokens_per_tile, tokens-first_token);
    const int lane = threadIdx.x;
    const int input_blocks = blocks_per_row >> input_block_shift;
    if (first_row >= rows) return;
    float partial[tokens_per_tile][rows_per_warp] = {};
    for (int b = lane; b < blocks_per_row; b += kWarpSize) {
        int4 w[rows_per_warp];
        float scale[rows_per_warp];
#pragma unroll
        for (int r = 0; r < rows_per_warp; ++r) {
            const size_t offset = static_cast<size_t>(first_row+r)*blocks_per_row+b;
            w[r] = packed[offset];
            scale[r] = __half2float(scales[offset]);
        }
#pragma unroll
        for (int t = 0; t < tokens_per_tile; ++t) {
            if (t < count) {
                const size_t input_block = static_cast<size_t>(first_token+t)*input_blocks+(b >> input_block_shift);
                const int4* a = reinterpret_cast<const int4*>(act_q)+2*input_block;
                const int4 a0=a[0], a1=a[1];
                const ActBlockInfo info=act_info[input_block];
#pragma unroll
                for (int r=0;r<rows_per_warp;++r) {
                    const int dot=pq4_block_dot(w[r],a0,a1);
                    partial[t][r] += scale[r]*(info.scale*static_cast<float>(dot)-kWeightCenter*info.scaled_sum);
                }
            }
        }
    }
#pragma unroll
    for (int t=0;t<tokens_per_tile;++t) {
        if (t < count) {
#pragma unroll
            for (int r=0;r<rows_per_warp;++r) {
                const float value=warp_sum(partial[t][r]);
                if(lane==0) out[static_cast<size_t>(first_token+t)*rows+first_row+r]=value;
            }
        }
    }
}
extern "C" {
// g4_gemm_small は 4 入力で重みを共有します。各出力の加算順は単一 GEMV と同じです。
// 起動: grid=(ceil(rows/16),ceil(tokens/4)), block=(32,4)。
__global__ void g4_gemm_small(const int4* __restrict__ packed, const __half* __restrict__ scales,
    const int8_t* __restrict__ act_q, const ActBlockInfo* __restrict__ act_info, float* __restrict__ out,
    int rows, int blocks_per_row, int tokens, int input_block_shift) {
    g4_gemm_small_body<4>(packed, scales, act_q, act_info, out, rows, blocks_per_row, tokens, input_block_shift);
}
// g4_gemm_small8 は 8 入力で重みを共有します。各出力の加算順は単一 GEMV と同じです。
// 起動: grid=(ceil(rows/16),ceil(tokens/8)), block=(32,4)。
__global__ void g4_gemm_small8(const int4* __restrict__ packed, const __half* __restrict__ scales,
    const int8_t* __restrict__ act_q, const ActBlockInfo* __restrict__ act_info, float* __restrict__ out,
    int rows, int blocks_per_row, int tokens, int input_block_shift) {
    g4_gemm_small_body<8>(packed, scales, act_q, act_info, out, rows, blocks_per_row, tokens, input_block_shift);
}
}

extern "C" {
// g4_mtp_hidden は最終正規化後の特徴量を FP32 で取り出します。量子化前の特徴を assistant へ渡します。
// 起動 1 block x 1024。例: n=3840、hidden は最後に受理した target の残差です。
__global__ void g4_mtp_hidden(const float* hidden, const float* weight, float eps, int n, float* out) {
    __shared__ float shared[32];
    float values[4];
    float squares=0.0f;
#pragma unroll
    for(int k=0;k<4;++k) {
        const int i=threadIdx.x+k*blockDim.x;
        values[k]=i<n ? hidden[i] : 0.0f;
        squares+=values[k]*values[k];
    }
    const float inverse=1.0f/sqrtf(block_sum(squares,shared)/n+eps);
#pragma unroll
    for(int k=0;k<4;++k) {
        const int i=threadIdx.x+k*blockDim.x;
        if(i<n) out[i]=weight[i]*(values[k]*inverse);
    }
}
// g4_mtp_concat は target の埋め込みと再帰特徴量を連結して int8 化します。
// 起動 ceil(2*hidden/256) x 256。例: hidden=3840 -> 入力 7680 次元。
__global__ void g4_mtp_concat(const unsigned char* packed, const __half* scales, const int* state,
    const float* recurrent, int hidden, float embedding_scale, int8_t* q, ActBlockInfo* info) {
    const int i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=2*hidden) return;
    const float value=i<hidden ? g4_embedding_value(packed,scales,state[0],hidden,i)*embedding_scale : recurrent[i-hidden];
    quantize_warp(value,i/kQuantBlock,q,info);
}
// g4_mtp_prepare は Q だけを正規化・回転します。target 共有 KV には書き込みません。
// 引数と grid は g4_attention_prepare と互換です。位置は全 draft ステップで固定します。
__global__ void g4_mtp_prepare(const float* qkv,const float* q_norm,const float* k_norm,
    const float* inverse_frequency,float eps,const int* state,int head_dim,int num_heads,int kv_heads,
    int value_offset,float* query,G4Key* keys,G4Value* values,int ring_size) {
    const int vector=(blockIdx.x*blockDim.x+threadIdx.x)/kWarpSize;
    if(vector<num_heads) g4_prepare_vector(qkv,q_norm,k_norm,inverse_frequency,eps,state[1],head_dim,
        num_heads,kv_heads,value_offset,query,keys,values,ring_size,vector);
}

// g4_mtp_local は位置 state[1] の Q から、その直前までの target KV を参照します。
__global__ void g4_mtp_local(const float* query,const G4Key* keys,const G4Value* values,const int* state,
    int window,int ring_size,int kv_heads,int heads_per_kv,float scale,float* partial) {
    const int prefix[4]={0,state[1]-1,0,0};
    g4_attention_decode_body<256,2,16>(query,keys,values,prefix,window>0?window-1:0,
        ring_size,kv_heads,heads_per_kv,scale,partial);
}
// g4_verify_local は grid.z の行を生成経路と同じ分割・加算順で計算します。
// partial_stride は 1 トークンの途中結果を置く FP32 要素数です。
__global__ void g4_verify_local(const float* query,const G4Key* keys,const G4Value* values,int start,
    int window,int ring_size,int kv_heads,int heads_per_kv,float scale,float* partial,int num_heads,int partial_stride) {
    const int row=blockIdx.z;
    const int state[4]={0,start+row,0,0};
    g4_attention_decode_body<256,2,16>(query+static_cast<size_t>(row)*num_heads*256,keys,values,state,
        window,ring_size,kv_heads,heads_per_kv,scale,partial+static_cast<size_t>(row)*partial_stride);
}

// g4_mtp_global は位置 state[1] の Q から、その直前までの target KV を参照します。
__global__ void g4_mtp_global(const float* query,const G4Key* keys,const G4Value* values,const int* state,
    int window,int ring_size,int kv_heads,int heads_per_kv,float scale,float* partial) {
    const int prefix[4]={0,state[1]-1,0,0};
    g4_attention_decode_body<512,4,8>(query,keys,values,prefix,window>0?window-1:0,
        ring_size,kv_heads,heads_per_kv,scale,partial);
}
// g4_verify_global は grid.z の行を生成経路と同じ分割・加算順で計算します。
// partial_stride は 1 トークンの途中結果を置く FP32 要素数です。
__global__ void g4_verify_global(const float* query,const G4Key* keys,const G4Value* values,int start,
    int window,int ring_size,int kv_heads,int heads_per_kv,float scale,float* partial,int num_heads,int partial_stride) {
    const int row=blockIdx.z;
    const int state[4]={0,start+row,0,0};
    g4_attention_decode_body<512,4,8>(query+static_cast<size_t>(row)*num_heads*512,keys,values,state,
        window,ring_size,kv_heads,heads_per_kv,scale,partial+static_cast<size_t>(row)*partial_stride);
}

// g4_attention_combine は通常の生成に使う部分和の合成です。引数と起動形状は従来と同じです。
__global__ void g4_attention_combine(const float* partial,const int* state,int window,int max_splits,
    int head_dim,int group_size,int num_heads,int8_t* q,ActBlockInfo* info) {
    g4_combine_body(partial,state,window,max_splits,head_dim,group_size,num_heads,q,info);
}
// g4_mtp_combine は assistant の可視範囲で部分和を合成します。Q の位置の直前までが確定 KV です。
__global__ void g4_mtp_combine(const float* partial,const int* state,int window,int max_splits,
    int head_dim,int group_size,int num_heads,int8_t* q,ActBlockInfo* info) {
    const int prefix[4]={0,state[1]-1,0,0};
    g4_combine_body(partial,prefix,window>0?window-1:0,max_splits,head_dim,group_size,num_heads,q,info);
}
// g4_verify_combine は grid.y の行を単一トークンと同じ式で合成・量子化します。
__global__ void g4_verify_combine(const float* partial,int start,int window,int max_splits,
    int head_dim,int group_size,int num_heads,int8_t* q,ActBlockInfo* info,int partial_stride) {
    const int row=blockIdx.y;
    const int state[4]={0,start+row,0,0};
    const int width=head_dim*num_heads;
    g4_combine_body(partial+static_cast<size_t>(row)*partial_stride,state,window,max_splits,head_dim,group_size,
        num_heads,q+static_cast<size_t>(row)*width,info+static_cast<size_t>(row)*(width/kQuantBlock));
}
}

extern "C" {
// g4_mtp_chain_start は既知の次トークンを検証列の先頭へ保存します。起動は 1 thread。
__global__ void g4_mtp_chain_start(const int* state, int* tokens) {
    tokens[0] = state[0];
}

// g4_mtp_chain_next は argmax を候補列へ保存し、次の assistant の入力にします。
// argmax_final_advance が進めた位置を戻し、全候補で同じ target prefix を参照します。
// 起動は 1 thread。例: index=1 は最初の候補を tokens[1] に書きます。
__global__ void g4_mtp_chain_next(int* state, int* tokens, int index) {
    tokens[index] = state[2];
    state[0] = state[2];
    state[1] -= 1;
}
}

// 128出力行×64トークンを処理し、Kを8量子化ブロックずつ共有メモリへ送る。
extern "C" __global__ void __launch_bounds__(kGemmThreads) g4_prefill_gemm(const int4* __restrict__ packed,
                                                         const __half* __restrict__ scales,
                                                         const int8_t* __restrict__ act_q,
                                                         const ActBlockInfo* __restrict__ act_info,
                                                         float* __restrict__ out, int rows, int blocks_per_row,
                                                         int tokens, int input_block_shift) {
    gemm_pq4_tile<128, 64, 8, 4, 8>(packed, scales, act_q, act_info, out, rows, blocks_per_row, tokens, input_block_shift);
}

#if !defined(G4_KEY_FP32) && !defined(G4_VALUE_FP32)
extern "C" {
// Graphの再生ごとに変わる検証開始位置をGPU上のdraft_state[1]から取得する。
__global__ void g4_attention_prepare_rows_dynamic(const float* __restrict__ qkv, const float* __restrict__ q_norm,
                                          const float* __restrict__ k_norm,
                                          const float* __restrict__ inverse_frequency, float eps, const int* draft_state,
                                          int head_dim, int num_heads, int kv_heads, int value_offset, int qkv_width,
                                          float* __restrict__ query_out, G4Key* __restrict__ key_cache,
                                          G4Value* __restrict__ value_cache, int ring_size) {
    const int vector = blockIdx.x * kG4Warps + threadIdx.x / kWarpSize;
    if (vector >= num_heads + 2 * kv_heads) return;
    const size_t row = blockIdx.y;
    g4_prepare_vector(qkv + row * qkv_width, q_norm, k_norm, inverse_frequency, eps,
                      draft_state[1] + static_cast<int>(row), head_dim, num_heads, kv_heads, value_offset,
                      query_out + row * num_heads * head_dim, key_cache, value_cache, ring_size, vector);
}
// Graphの再生ごとに変わる検証開始位置をGPU上のdraft_state[1]から取得する。
__global__ void g4_verify_local_dynamic(const float* query,const G4Key* keys,const G4Value* values,const int* draft_state,
    int window,int ring_size,int kv_heads,int heads_per_kv,float scale,float* partial,int num_heads,int partial_stride) {
    const int row=blockIdx.z;
    const int state[4]={0,draft_state[1]+row,0,0};
    g4_attention_decode_body<256,2,16>(query+static_cast<size_t>(row)*num_heads*256,keys,values,state,
        window,ring_size,kv_heads,heads_per_kv,scale,partial+static_cast<size_t>(row)*partial_stride);
}
// Graphの再生ごとに変わる検証開始位置をGPU上のdraft_state[1]から取得する。
__global__ void g4_verify_global_dynamic(const float* query,const G4Key* keys,const G4Value* values,const int* draft_state,
    int window,int ring_size,int kv_heads,int heads_per_kv,float scale,float* partial,int num_heads,int partial_stride) {
    const int row=blockIdx.z;
    const int state[4]={0,draft_state[1]+row,0,0};
    g4_attention_decode_body<512,4,8>(query+static_cast<size_t>(row)*num_heads*512,keys,values,state,
        window,ring_size,kv_heads,heads_per_kv,scale,partial+static_cast<size_t>(row)*partial_stride);
}
// Graphの再生ごとに変わる検証開始位置をGPU上のdraft_state[1]から取得する。
__global__ void g4_verify_combine_dynamic(const float* partial,const int* draft_state,int window,int max_splits,
    int head_dim,int group_size,int num_heads,int8_t* q,ActBlockInfo* info,int partial_stride) {
    const int row=blockIdx.y;
    const int state[4]={0,draft_state[1]+row,0,0};
    const int width=head_dim*num_heads;
    g4_combine_body(partial+static_cast<size_t>(row)*partial_stride,state,window,max_splits,head_dim,group_size,
        num_heads,q+static_cast<size_t>(row)*width,info+static_cast<size_t>(row)*(width/kQuantBlock));
}
// 固定3候補の連続一致長と各位置のIDをGPUでまとめる。1スレッドで実行する。
__global__ void g4_accept_three(const int* inputs, const int* states, int stride, int argmax_index, int* result) {
    int accepted = 0;
    for (int row = 0; row < 4; ++row) {
        result[row] = inputs[row];
        result[4 + row] = states[row * stride + argmax_index];
    }
    while (accepted < 3 && inputs[accepted + 1] == states[accepted * stride + argmax_index]) {
        ++accepted;
    }
    result[8] = accepted;
}
}
#endif
