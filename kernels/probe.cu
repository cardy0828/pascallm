// probe.cu
//
// M1a（Go から purego 経由で CUDA を呼ぶ試作）で使う確認用カーネルです。
// Go 側から名前で探せるよう、すべて extern "C" で公開します。
// ビルドは kernels/build.sh（sm_61 の cubin を internal/kernels/ に出力）で行います。

#include <cstdint>

extern "C" {

// y[i] = y[i] + a * x[i] を n 要素について計算します。
// 引数の受け渡し（ポインター 2 個・float・int）が正しいかを確かめるためのカーネルです。
// 例: x = {1, 2}, y = {1, 1}, a = 2 -> y = {3, 5}
__global__ void axpy_f32(float* __restrict__ y, const float* __restrict__ x, float a, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        y[index] = y[index] + a * x[index];
    }
}

// out[i] = DP4A(a[i], b[i], 0) を n 要素について計算します。
// a[i]・b[i] は int8 を 4 個詰めた 32 ビット値で、結果はその 4 組の積の和です。
// cubin が sm_61 の DP4A 命令付きで動くことを確かめるためのカーネルです。
// 例: a = 0x01020304, b = 0x01010101 -> 1 + 2 + 3 + 4 = 10
__global__ void dot4_i8(const int32_t* __restrict__ a, const int32_t* __restrict__ b,
                        int32_t* __restrict__ out, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        out[index] = __dp4a(a[index], b[index], 0);
    }
}

// data[i] に 1 を加えます。起動コストの計測と、CUDA Graph の再生回数の確認に使います。
// 例: 同じ data に 600 回起動すると、各要素が 600 増える。
__global__ void increment_f32(float* __restrict__ data, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < n) {
        data[index] += 1.0f;
    }
}

}  // extern "C"
