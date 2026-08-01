#include <assert.h>
#include <math.h>
#include <stddef.h>

float rsqrtf(float value);

static int close_enough(float actual, double expected, double relative)
{
    double error = (double)actual - expected;
    double scale = expected < 0.0 ? -expected : expected;

    if (error < 0.0)
        error = -error;
    if (scale < 1.0)
        scale = 1.0;
    return error <= relative * scale;
}

int main(void)
{
    static const float exp_values[] = {
        -80.0f, -10.0f, -1.0f, -0.1f, 0.0f,
        0.1f, 1.0f, 5.0f, 10.0f, 80.0f,
    };
    static const float log_values[] = {
        1.0e-30f, 0.01f, 0.1f, 0.5f, 1.0f,
        2.0f, 10.0f, 1000.0f, 1.0e30f,
    };
    size_t i;

    for (i = 0; i < sizeof(exp_values) / sizeof(exp_values[0]); ++i) {
        float value = exp_values[i];

        assert(close_enough(expf(value), exp((double)value), 3.0e-6));
    }
    for (i = 0; i < sizeof(log_values) / sizeof(log_values[0]); ++i) {
        float value = log_values[i];

        assert(close_enough(logf(value), log((double)value), 3.0e-6));
        assert(close_enough(sqrtf(value), sqrt((double)value), 2.0e-6));
        assert(close_enough(rsqrtf(value), 1.0 / sqrt((double)value), 3.0e-6));
    }

    assert(isinf(expf(1000.0f)) && expf(1000.0f) > 0.0f);
    assert(expf(-1000.0f) == 0.0f);
    assert(isinf(logf(0.0f)) && logf(0.0f) < 0.0f);
    assert(isnan(logf(-1.0f)));
    assert(signbit(fminf(0.0f, -0.0f)));
    assert(!signbit(fmaxf(0.0f, -0.0f)));
    return 0;
}
