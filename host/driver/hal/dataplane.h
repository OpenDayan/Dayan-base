#ifndef NPU_HOST_HAL_DATAPLANE_H
#define NPU_HOST_HAL_DATAPLANE_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/*
 * Stable boundary between dayan_server and libDayanRuntime's data worker.
 * The ABI deliberately contains no Device address-space, Cluster or Core
 * model.  A provider interprets the opaque export and token payloads.
 */
#define NPU_HAL_DATA_EXPORT_VERSION  1U
#define NPU_HAL_DATA_TOKEN_VERSION   1U
#define NPU_HAL_DATA_MAX_HANDLES     64U

/* Seekable H2D/D2H file descriptors, used by the current FUSE simulator. */
#define NPU_HAL_DATA_PROVIDER_FD_OFFSET 1U

typedef struct npu_hal_data_client npu_hal_data_client_t;

typedef enum npu_hal_data_direction {
    NPU_HAL_DATA_HOST_TO_DEVICE = 1,
    NPU_HAL_DATA_DEVICE_TO_HOST = 2,
} npu_hal_data_direction_t;

/*
 * Sent once with SCM_RIGHTS.  handle_count is generic; handle ordering and
 * opaque fields belong exclusively to the selected provider implementation.
 */
typedef struct npu_hal_data_export {
    uint32_t version;
    uint32_t provider;
    uint32_t handle_count;
    uint32_t flags;
    uint64_t opaque[4];
} npu_hal_data_export_t;

/*
 * Issued for exactly one transfer.  Runtime treats every field except size as
 * opaque and returns the descriptor unchanged when completing the transfer.
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

int npu_hal_data_client_import(npu_hal_data_client_t **client_out,
                               const npu_hal_data_export_t *exported,
                               int *handles, size_t handle_count);
void npu_hal_data_client_close(npu_hal_data_client_t *client);
int npu_hal_data_client_transfer(npu_hal_data_client_t *client,
                                 const npu_hal_data_token_t *token,
                                 void *host_buffer, size_t size);

#ifdef __cplusplus
}
#endif

#endif
