#include <dayan_runtime.h>

static_assert(static_cast<uint32_t>(NPUTaskMode::kSingleCore) == 1);
static_assert(static_cast<uint32_t>(NPUTaskMode::kFourCore) == 4);
static_assert(sizeof(NPUTaskDim3) == 12);
static_assert(sizeof(NPULaunchConfig) == 16);

int public_headers_compile()
{
    NPULaunchConfig config{{4, 2, 3}, NPUTaskMode::kFourCore};
    return config.task_dim.x == 4 ? 0 : 1;
}
