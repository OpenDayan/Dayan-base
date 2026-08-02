#ifndef NPU_HOST_DRIVER_H
#define NPU_HOST_DRIVER_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define NPU_DRIVER_OPTIONS_VERSION 1U
#define NPU_DRIVER_MAX_HARTS       32U

typedef struct npu_driver_options {
    uint32_t version;
    uint32_t device_index;
    const char *device_prefix;
    uint64_t memory_address_bias;
} npu_driver_options_t;

typedef struct npu_device {
    int h2c_fd;
    int c2h_fd;
    int user_fd;
    int event_fd;
    uint64_t memory_address_bias;
    uint32_t device_index;
    uint32_t opened;
} npu_device_t;

void npu_driver_options_init(npu_driver_options_t *options);
int npu_device_open(npu_device_t *device,
                    const npu_driver_options_t *options);
void npu_device_close(npu_device_t *device);

int npu_device_read(npu_device_t *device, uint64_t address,
                    void *data, size_t size);
int npu_device_write(npu_device_t *device, uint64_t address,
                     const void *data, size_t size);
int npu_device_read_control(npu_device_t *device, uint64_t offset,
                            void *data, size_t size);
int npu_device_write_control(npu_device_t *device, uint64_t offset,
                             const void *data, size_t size);

int npu_device_read_irq_status(npu_device_t *device, uint32_t *status);
int npu_device_clear_irq(npu_device_t *device, uint32_t mask);
int npu_device_read_irq_mailbox(npu_device_t *device, uint32_t hart,
                                uint64_t *command);
int npu_device_submit(npu_device_t *device, uint32_t hart,
                      uint64_t command);
int npu_device_wait_msip_clear(npu_device_t *device, uint32_t hart,
                               uint32_t timeout_ms);
int npu_device_boot(npu_device_t *device, uint32_t hart_mask,
                    uint32_t hart_count, uint64_t entry,
                    uint32_t timeout_ms);

#ifdef __cplusplus
}
#endif

#endif
