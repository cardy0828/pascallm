// Package numeric は、BF16・FP16 と FP32 の相互変換など、数値形式の小さな道具をまとめます。
package numeric

import "math"

// BF16ToFloat32 は、bfloat16 のビット列を float32 に変換します。
// bfloat16 は float32 の上位 16 ビットそのものなので、16 ビット左へずらすだけで正確に変換できます。
// 例: 0x3F80 -> 1.0, 0xC000 -> -2.0
func BF16ToFloat32(bits uint16) float32 {
	return math.Float32frombits(uint32(bits) << 16)
}

// float16 と float32 のビット配置に関する定数です。
const (
	float32ExponentBias = 127
	float16ExponentBias = 15
	float16MaxExponent  = 31 // 指数部が全ビット 1（無限大または NaN）
	float32MantissaBits = 23
	float16MantissaBits = 10
	mantissaShift       = float32MantissaBits - float16MantissaBits // 13
)

// Float16ToFloat32 は、IEEE 754 半精度（FP16）のビット列を float32 に変換します（誤差なし）。
// 例: 0x3C00 -> 1.0, 0xC000 -> -2.0, 0x0001 -> 5.96e-8（最小の非正規化数）
func Float16ToFloat32(bits uint16) float32 {
	sign := uint32(bits>>15) & 1
	exponent := int32(bits>>float16MantissaBits) & 0x1F
	mantissa := uint32(bits) & 0x3FF

	switch {
	case exponent == 0 && mantissa == 0:
		return math.Float32frombits(sign << 31)
	case exponent == 0:
		// 非正規化数: 仮数を正規化しながら指数を下げる。
		for mantissa&0x400 == 0 {
			mantissa <<= 1
			exponent--
		}
		exponent++
		mantissa &= 0x3FF
	case exponent == float16MaxExponent:
		// 無限大または NaN。
		return math.Float32frombits(sign<<31 | 0xFF<<float32MantissaBits | mantissa<<mantissaShift)
	}
	float32Exponent := uint32(exponent - float16ExponentBias + float32ExponentBias)
	return math.Float32frombits(sign<<31 | float32Exponent<<float32MantissaBits | mantissa<<mantissaShift)
}

// Float32ToFloat16 は、float32 を IEEE 754 半精度（FP16）のビット列に変換します。
// 丸めは最近接偶数丸め（CUDA の __float2half_rn と同じ）。範囲外の値は無限大になります。
// 例: 1.0 -> 0x3C00, 65504 -> 0x7BFF（FP16 の最大値）, 1e-8 -> 0x0000
func Float32ToFloat16(value float32) uint16 {
	bits := math.Float32bits(value)
	sign := uint16(bits>>16) & 0x8000
	exponent := int32(bits>>float32MantissaBits) & 0xFF
	mantissa := bits & 0x7FFFFF

	if exponent == 0xFF {
		// 無限大または NaN（NaN は仮数の最上位ビットを立てて静かな NaN にする）。
		if mantissa != 0 {
			return sign | 0x7E00
		}
		return sign | 0x7C00
	}
	halfExponent := exponent - float32ExponentBias + float16ExponentBias
	if halfExponent >= float16MaxExponent {
		return sign | 0x7C00
	}
	if halfExponent <= 0 {
		// 非正規化数またはゼロ。暗黙の 1 を仮数に足してから、必要な分だけ右へずらす。
		if halfExponent < -float16MantissaBits {
			return sign
		}
		mantissa |= 1 << float32MantissaBits
		shift := uint32(mantissaShift + 1 - halfExponent)
		return sign | uint16(roundShiftRightEven(mantissa, shift))
	}
	// 正規化数。丸めで仮数が桁上がりすると指数も 1 増えるが、ビット列を足し算すれば自然に正しくなる。
	result := uint32(halfExponent)<<float16MantissaBits | roundShiftRightEven(mantissa, mantissaShift)
	return sign | uint16(result)
}

// roundShiftRightEven は、value を shift ビット右へずらし、はみ出したビットで最近接偶数丸めをします。
// 例: roundShiftRightEven(0b1011, 2) = 0b11（0.75 を切り上げ）, roundShiftRightEven(0b1010, 2) = 0b10（0.5 は偶数へ）
func roundShiftRightEven(value uint32, shift uint32) uint32 {
	if shift == 0 {
		return value
	}
	if shift > 31 {
		return 0
	}
	kept := value >> shift
	remainder := value & (1<<shift - 1)
	half := uint32(1) << (shift - 1)
	if remainder > half || (remainder == half && kept&1 == 1) {
		kept++
	}
	return kept
}

// RoundFloat32ToFloat16 は、float32 を FP16 に丸めた値を float32 で返します。
// 例: RoundFloat32ToFloat16(0.1) = 0.0999755859375
func RoundFloat32ToFloat16(value float32) float32 {
	return Float16ToFloat32(Float32ToFloat16(value))
}
