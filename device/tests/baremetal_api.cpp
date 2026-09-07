#include <dayan_test.h>

static void entry() {}

DAYAN_TEST(cxx_test, {entry, {}, {}, 0, "cpp"});

static_assert(sizeof(dayan_test_desc_t) == 128, "C++ test descriptor ABI");
