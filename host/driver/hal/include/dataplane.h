#ifndef NPU_HOST_HAL_DATAPLANE_H
#define NPU_HOST_HAL_DATAPLANE_H

/* Public HAL data-plane ABI shared by Server and Runtime. */

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Stable boundary between dayan_server and libDayanRuntime's transfer thread.
 * Provider code runs in the Runtime process and accesses imported device
 * handles directly. It never sends the transfer payload through the Server
 * transport. A provider alone interprets the opaque export and token fields.
 */
#define NPU_HAL_DATA_EXPORT_VERSION  1U
#define NPU_HAL_DATA_TOKEN_VERSION   1U
#define NPU_HAL_DATA_MAX_HANDLES     64U

#define NPU_HAL_DATA_PROVIDER_XILINX_DMA 1U
#define NPU_HAL_DATA_PROVIDER_XDMA NPU_HAL_DATA_PROVIDER_XILINX_DMA
#define NPU_HAL_DATA_PROVIDER_SOC_MMAP   2U
#define NPU_HAL_DATA_PROVIDER_INPROCESS  3U

typedef struct npu_hal_data_plane npu_hal_data_plane_t;

typedef enum npu_hal_data_direction {
    NPU_HAL_DATA_HOST_TO_DEVICE = 1,
    NPU_HAL_DATA_DEVICE_TO_HOST = 2,
} npu_hal_data_direction_t;

/*
 * Describes the provider resources exported during Runtime setup. The generic
 * transport may pass the handles to another process; the provider itself does
 * not depend on that transport. Handle ordering and opaque fields are private
 * to the selected provider implementation.
 */
typedef struct npu_hal_data_export {
    uint32_t version;
    uint32_t provider;
    uint32_t handle_count;
    uint32_t flags;
    uint64_t opaque[4];
} npu_hal_data_export_t;

/*
 * Issued for exactly one transfer. Runtime treats provider-specific fields as
 * opaque and returns the complete token unchanged with the transfer result.
 */
typedef struct npu_hal_data_token {
    uint32_t version;
    uint32_t provider;
    uint32_t direction;
    uint32_t flags;
    uint64_t size;
    uint64_t opaque[4];
} npu_hal_data_token_t;

#ifdef __cplusplus
static_assert(sizeof(npu_hal_data_export_t) == 48,
              "HAL data export ABI");
static_assert(sizeof(npu_hal_data_token_t) == 56,
              "HAL data token ABI");
#else
_Static_assert(sizeof(npu_hal_data_export_t) == 48,
               "HAL data export ABI");
_Static_assert(sizeof(npu_hal_data_token_t) == 56,
               "HAL data token ABI");
#endif

/* Runtime-side API. Import takes ownership of every accepted device handle. */
int npu_hal_data_plane_import(npu_hal_data_plane_t **data_plane_out,
                              const npu_hal_data_export_t *exported,
                              int *handles, size_t handle_count);
void npu_hal_data_plane_close(npu_hal_data_plane_t *data_plane);
int npu_hal_data_plane_transfer(npu_hal_data_plane_t *data_plane,
                                const npu_hal_data_token_t *token,
                                void *host_buffer, size_t size);

#ifdef __cplusplus
}
#endif

#endif
