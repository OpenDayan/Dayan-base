#ifndef DAYAN_QEMU_ASSERT_H
#define DAYAN_QEMU_ASSERT_H

#ifdef __cplusplus
extern "C" {
#endif

__attribute__((noreturn))
void dayan_test_assert_fail(const char *expression, const char *file,
                            unsigned line);

#ifdef __cplusplus
}
#endif

#define assert(expression)                                                   \
    ((expression) ? (void)0 :                                                \
     dayan_test_assert_fail(#expression, __FILE__, __LINE__))

#endif
