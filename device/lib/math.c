#include <stdint.h>

#include "math.h"

/*
 * Small single-precision routines for rv64imaf.  The range reduction used by
 * expf/logf use an fdlibm-style range reduction, but keep every operation in
 * float so this file never needs soft-double helpers.
 */

static uint32_t float_bits(float value)
{
    union {
        float value;
        uint32_t bits;
    } data = { value };

    return data.bits;
}

static float bits_float(uint32_t bits)
{
    union {
        uint32_t bits;
        float value;
    } data = { bits };

    return data.value;
}

float fabsf(float value)
{
    return bits_float(float_bits(value) & 0x7fffffffU);
}

float copysignf(float value, float sign)
{
    return bits_float((float_bits(value) & 0x7fffffffU) |
                      (float_bits(sign) & 0x80000000U));
}

float fminf(float left, float right)
{
    if (isnan(left))
        return right;
    if (isnan(right))
        return left;
    if (left == right)
        return bits_float(float_bits(left) | float_bits(right));
    return left < right ? left : right;
}

float fmaxf(float left, float right)
{
    if (isnan(left))
        return right;
    if (isnan(right))
        return left;
    if (left == right)
        return bits_float(float_bits(left) & float_bits(right));
    return left > right ? left : right;
}

float sqrtf(float value)
{
    float result;

    __asm__ volatile("fsqrt.s %0, %1" : "=f"(result) : "f"(value));
    return result;
}

float rsqrtf(float value)
{
    uint32_t bits = float_bits(value);
    float half;
    float result;

    if ((bits & 0x7fffffffU) == 0)
        return copysignf(INFINITY, value);
    if (bits & 0x80000000U)
        return NAN;
    if ((bits & 0x7f800000U) == 0x7f800000U)
        return bits & 0x007fffffU ? value : 0.0f;

    half = value * 0.5f;
    result = bits_float(0x5f375a86U - (bits >> 1));
    result *= 1.5f - half * result * result;
    result *= 1.5f - half * result * result;
    result *= 1.5f - half * result * result;
    return result;
}

static float scale_pow2(float value, int exponent)
{
    if (exponent > 127)
        return INFINITY;
    if (exponent < -149)
        return 0.0f;
    if (exponent >= -126)
        return value * bits_float((uint32_t)(exponent + 127) << 23);

    return value * bits_float(1U << 23) *
           bits_float((uint32_t)(exponent + 253) << 23);
}

float expf(float value)
{
    const float inv_ln2 = 0x1.715476p+0f;
    const float ln2_hi = 0x1.62e400p-1f;
    const float ln2_lo = 0x1.7f7d1cp-20f;
    float scaled;
    float reduced;
    float square;
    float polynomial;
    int exponent;

    if (isnan(value))
        return value;
    if (value > 0x1.62e42ep+6f)
        return INFINITY;
    if (value < -0x1.9fe36ap+6f)
        return 0.0f;

    scaled = value * inv_ln2;
    exponent = (int)(scaled + (scaled < 0.0f ? -0.5f : 0.5f));
    reduced = (value - (float)exponent * ln2_hi) -
              (float)exponent * ln2_lo;
    square = reduced * reduced;
    polynomial = 1.0f + reduced + square *
        (0.5f + reduced *
         (0.1666666716f + reduced *
          (0.0416666679f + reduced * 0.0083333338f)));
    return scale_pow2(polynomial, exponent);
}

float logf(float value)
{
    const float ln2_hi = 0x1.62e400p-1f;
    const float ln2_lo = 0x1.7f7d1cp-20f;
    uint32_t bits = float_bits(value);
    int exponent;
    float mantissa;
    float ratio;
    float square;
    float series;

    if ((bits & 0x7fffffffU) == 0)
        return -INFINITY;
    if (bits & 0x80000000U)
        return NAN;
    if ((bits & 0x7f800000U) == 0x7f800000U)
        return value;

    if ((bits & 0x7f800000U) == 0) {
        value *= 0x1p23f;
        bits = float_bits(value);
        exponent = -23;
    } else {
        exponent = 0;
    }

    exponent += (int)(bits >> 23) - 127;
    mantissa = bits_float((bits & 0x007fffffU) | 0x3f800000U);
    if (mantissa > 0x1.6a09e6p+0f) {
        mantissa *= 0.5f;
        ++exponent;
    }

    ratio = (mantissa - 1.0f) / (mantissa + 1.0f);
    square = ratio * ratio;
    series = ratio * (2.0f + square *
        (0.6666666865f + square *
         (0.4f + square *
          (0.2857142985f + square * 0.2222222239f))));
    return ((float)exponent * ln2_hi + series) +
           (float)exponent * ln2_lo;
}
