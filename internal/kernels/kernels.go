// Package kernels は、nvcc で事前にコンパイルした cubin を Go のバイナリに埋め込みます。
//
// cubin は kernels/build.sh で再生成します。このパッケージ自体は cgo も nvcc も使いません。
package kernels

import _ "embed"

// ProbeSM61 は kernels/probe.cu を sm_61 向けにコンパイルした cubin です。
// 含まれるカーネル: axpy_f32, dot4_i8, increment_f32
//
//go:embed probe_sm_61.cubin
var ProbeSM61 []byte

// LFM3SM61 は kernels/lfm3.cu を sm_61 向けにコンパイルした cubin です（LFM3 の 1 トークン生成用）。
// 含まれるカーネル: embed_lookup, residual_norm_quantize, gemv_pq4, short_conv_quantize,
// qk_norm_rope_cache, attention_decode, silu_mul_quantize, argmax_partial, argmax_final_advance
//
//go:embed lfm3_sm_61.cubin
var LFM3SM61 []byte

// Gemma4SM61 は kernels/gemma4.cu を sm_61 向けにコンパイルした cubin です（Gemma 4 の生成と入力処理用）。
// gemma4.cu は lfm3.cu を取り込むため、共通のカーネル（gemv_pq4・gemm_pq4・argmax_partial・candidates_final など）も入っています。
// 含まれる Gemma 4 用のカーネル: g4_embed_pq4, g4_residual_norm_quantize, g4_attention_prepare, g4_attention_local,
// g4_attention_global, g4_attention_combine, g4_gelu_mul_quantize, g4_softcap, g4_candidates_partial と、入力処理用の *_rows
//
//go:embed gemma4_sm_61.cubin
var Gemma4SM61 []byte

// Gemma4K32SM61 は K FP32 / V FP16 の比較用 cubin です。
//
//go:embed gemma4_k32_sm_61.cubin
var Gemma4K32SM61 []byte

// Gemma4KV32SM61 は品質の参照分布を測る K/V FP32 の診断用 cubin です。
//
//go:embed gemma4_kv32_sm_61.cubin
var Gemma4KV32SM61 []byte
