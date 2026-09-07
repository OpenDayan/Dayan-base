#include <dayan_test.h>
#include <stdio.h>

static void fault(void)
{
    __asm__ volatile(".word 0");
}

static void after_fault(void)
{
    printf("continued after fault\n");
}

DAYAN_TEST(fault_test, {.entry = fault, .name = "fault"});
DAYAN_TEST(after_fault_test, {.entry = after_fault, .name = "after-fault"});
