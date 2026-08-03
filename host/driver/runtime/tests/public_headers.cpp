#include <dayan_runtime.h>

#include "service.h"

static_assert(static_cast<uint32_t>(NPUTaskMode::kSingleCore) == 1);
static_assert(static_cast<uint32_t>(NPUTaskMode::kFourCore) == 4);
static_assert(sizeof(NPUTaskDim3) == 12);
static_assert(sizeof(NPULaunchConfig) == 16);
static_assert(NPU_PROFILE_PROC_RUNNING == 0);
static_assert(NPU_PROFILE_PROC_KILL_WAIT_PRODUCE == 7);

int public_headers_compile()
{
    NPULaunchConfig config{{4, 2, 3}, NPUTaskMode::kFourCore};
    npu_profile_event_t profile_event{};
    npu_profile_batch_info_t profile_batch{};
    npu_service_proc_snapshot_t snapshot{};
    npu_service_kill_result_t killed{};
    return config.task_dim.x == 4 && snapshot.count == 0 &&
                   killed.waiting_produce == 0 &&
                   profile_event.state == NPU_PROFILE_PROC_RUNNING &&
                   profile_batch.count == 0 ?
        0 : 1;
}
