#include <dayan_runtime.h>

#include "service.h"

static_assert(static_cast<uint32_t>(NPUTaskMode::kSingleCore) == 1);
static_assert(static_cast<uint32_t>(NPUTaskMode::kFourCore) == 4);
static_assert(sizeof(NPUTaskDim3) == 12);
static_assert(sizeof(NPULaunchConfig) == 16);
static_assert(NPU_PROC_NAME_SIZE == 16);
static_assert(sizeof(npu_profile_record_t) == 8);
static_assert(sizeof(npu_profile_slot_define_t) == 24);
static_assert(NPU_SERVICE_PROC_NAME_SIZE == NPU_PROC_NAME_SIZE);
static_assert(DAYAN_KERNEL_RESOURCE_DESC_VERSION == 3);
static_assert(sizeof(dayan_kernel_resource_desc_t) == 40);
static_assert(offsetof(dayan_kernel_resource_desc_t, flags) == 12);
static_assert(offsetof(dayan_kernel_resource_desc_t, max_tasks_per_core) == 10);
static_assert(NPU_KERNEL_LOCKED_TASK == NPU_PACKAGE_KERNEL_LOCKED_TASK);
static_assert(NPU_PROFILE_PROC_RUNNING == 0);
static_assert(NPU_PROFILE_PROC_KILL_WAIT_PRODUCE == 7);

int public_headers_compile()
{
    NPULaunchConfig config{{4, 2, 3}, NPUTaskMode::kFourCore};
    npu_profile_record_t profile_record{};
    npu_profile_batch_info_t profile_batch{};
    npu_service_proc_snapshot_t snapshot{};
    npu_service_kill_result_t killed{};
    return config.task_dim.x == 4 && snapshot.count == 0 &&
                   killed.waiting_produce == 0 &&
                   profile_record.value == 0 &&
                   profile_batch.byte_count == 0 ?
        0 : 1;
}
