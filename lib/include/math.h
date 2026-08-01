#ifndef NPU_MATH_H
#define NPU_MATH_H

#define INFINITY (__builtin_inff())
#define NAN      (__builtin_nanf(""))
#define HUGE_VALF INFINITY

#define isnan(value)    __builtin_isnan(value)
#define isinf(value)    __builtin_isinf(value)
#define isfinite(value) __builtin_isfinite(value)
#define signbit(value)  __builtin_signbit(value)

float fabsf(float value);
float copysignf(float value, float sign);
float fminf(float left, float right);
float fmaxf(float left, float right);
float sqrtf(float value);
float rsqrtf(float value);
float expf(float value);
float logf(float value);

#endif
