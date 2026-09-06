#ifndef TEST_SECTION
#define TEST_SECTION ".rodata.test"
#endif
#ifndef TEST_QUALIFIER
#define TEST_QUALIFIER const
#endif

TEST_QUALIFIER unsigned long test_global
    __attribute__((used, section(TEST_SECTION))) = 0;

void _start(void)
{
}
