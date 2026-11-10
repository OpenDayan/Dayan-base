/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include <assert.h>
#include <math.h>
#include <stddef.h>

float rsqrtf(float value);

static int close_enough(float actual, float expected, float relative)
{
    float error = actual - expected;
    float scale = expected < 0.0f ? -expected : expected;

    if (error < 0.0f)
        error = -error;
    if (scale < 1.0f)
        scale = 1.0f;
    return error <= relative * scale;
}

int main(void)
{
    static const float exp_values[] = {
        -80.0f, -10.0f, -1.0f, -0.1f, 0.0f,
        0.1f, 1.0f, 5.0f, 10.0f, 80.0f,
    };
    static const float exp_expected[] = {
        1.8048514e-35f, 4.5399931e-5f, 0.36787945f, 0.90483743f,
        1.0f, 1.1051702f, 2.7182817f, 148.41316f, 22026.466f,
        5.5406225e34f,
    };
    static const float log_values[] = {
        1.0e-30f, 0.01f, 0.1f, 0.5f, 1.0f,
        2.0f, 10.0f, 1000.0f, 1.0e30f,
    };
    static const float log_expected[] = {
        -69.077553f, -4.6051702f, -2.3025851f, -0.69314718f, 0.0f,
        0.69314718f, 2.3025851f, 6.9077554f, 69.077553f,
    };
    static const float sqrt_expected[] = {
        1.0e-15f, 0.1f, 0.31622776f, 0.70710677f, 1.0f,
        1.4142135f, 3.1622777f, 31.622776f, 1.0e15f,
    };
    static const float rsqrt_expected[] = {
        1.0e15f, 10.0f, 3.1622777f, 1.4142135f, 1.0f,
        0.70710677f, 0.31622776f, 0.031622775f, 1.0e-15f,
    };
    size_t i;

    for (i = 0; i < sizeof(exp_values) / sizeof(exp_values[0]); ++i) {
        float value = exp_values[i];

        assert(close_enough(expf(value), exp_expected[i], 3.0e-6f));
    }
    for (i = 0; i < sizeof(log_values) / sizeof(log_values[0]); ++i) {
        float value = log_values[i];

        assert(close_enough(logf(value), log_expected[i], 3.0e-6f));
        assert(close_enough(sqrtf(value), sqrt_expected[i], 2.0e-6f));
        assert(close_enough(rsqrtf(value), rsqrt_expected[i], 3.0e-6f));
    }

    assert(isinf(expf(1000.0f)) && expf(1000.0f) > 0.0f);
    assert(expf(-1000.0f) == 0.0f);
    assert(isinf(logf(0.0f)) && logf(0.0f) < 0.0f);
    assert(isnan(logf(-1.0f)));
    assert(signbit(fminf(0.0f, -0.0f)));
    assert(!signbit(fmaxf(0.0f, -0.0f)));
    return 0;
}
