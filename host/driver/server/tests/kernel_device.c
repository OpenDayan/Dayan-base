#include <stdint.h>

extern void test_api(void);

uint64_t loader_global = 9;

void loader_kernel(uint64_t *output)
{
    test_api();
    *output = loader_global;
}
