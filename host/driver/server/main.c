#define _POSIX_C_SOURCE 200809L

#include "server.h"
#include "backend/xilinx_dma/npu_hal_xilinx_dma.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#ifndef DAYAN_DEFAULT_FIRMWARE
#define DAYAN_DEFAULT_FIRMWARE "build/device/npu-os.elf"
#endif
#ifndef DAYAN_DEFAULT_API_ELF
#define DAYAN_DEFAULT_API_ELF "build/device/npu-os-api.elf"
#endif

static volatile sig_atomic_t stop_requested;

static void stop_signal(int signal_number)
{
    (void)signal_number;
    stop_requested = 1;
}

static int parse_u32(const char *value, uint32_t *result)
{
    char *end;
    unsigned long parsed;

    errno = 0;
    parsed = strtoul(value, &end, 0);
    if (errno || !*value || value[0] == '-' || *end || parsed > UINT32_MAX)
        return -EINVAL;
    *result = (uint32_t)parsed;
    return 0;
}

static int parse_u64(const char *value, uint64_t *result)
{
    char *end;
    unsigned long long parsed;

    errno = 0;
    parsed = strtoull(value, &end, 0);
    if (errno || !*value || value[0] == '-' || *end)
        return -EINVAL;
    *result = (uint64_t)parsed;
    return 0;
}

static int parse_dma_profile(const char *value,
                              npu_hal_xilinx_dma_profile_t *profile)
{
    if (strcmp(value, "simulator") == 0) {
        *profile = NPU_HAL_XILINX_DMA_PROFILE_SIMULATOR;
        return 0;
    }
    if (strcmp(value, "fpga") == 0) {
        *profile = NPU_HAL_XILINX_DMA_PROFILE_FPGA;
        return 0;
    }
    return -EINVAL;
}

static void usage(FILE *stream, const char *program)
{
    fprintf(stream,
            "usage: %s [options]\n"
            "  --socket PATH           Unix socket path\n"
            "  --firmware PATH         BL1 ELF\n"
            "  --api-elf PATH          public BL1 API ELF\n"
            "  --backend NAME          xilinx-dma\n"
            "  --dma-driver NAME       xdma (default) or qdma\n"
            "  --dma-profile NAME      simulator (default) or fpga\n"
            "  --dma-device-dir PATH   override profile directory\n"
            "  --dma-device-name NAME  xdma0 or qdma01000, for one Cluster\n"
            "  --dma-mmio-offset N     override profile low-address offset\n"
            "  --dma-queue N           QDMA MM queue (default 0)\n"
            "  --dma-user-irq N        QDMA user IRQ (default 0)\n"
            "  --dma-channel N         XDMA H2C/C2H channel (default 0)\n"
            "  --dma-event N           XDMA event (default 0)\n"
            "  --dma-completion MODE   irq (required node) or polling (1 us sleep)\n"
            "  --xdma-profile/--xdma-device-dir/--xdma-mmio-offset: legacy aliases\n"
            "  --device-index N        first XDMA device index\n"
            "  --cluster-count N       number of Clusters\n"
            "  --cluster-stride N      XDMA index stride per Cluster\n"
            "  --cluster-mask MASK     enabled Cluster mask\n"
            "  --core-mask MASK        enabled Core mask within Cluster\n"
            "  --timeout-ms N          Device operation timeout in milliseconds\n"
            "  --normal-memory-size-mib N  default Normal memory size in MiB\n"
            "  --enable-mmu 0|1        route Device addresses through the ClusterMMU\n"
            "  --help                  show this text\n",
            program);
}

int main(int argc, char **argv)
{
    dayan_server_options_t options;
    npu_hal_xilinx_dma_options_t dma;
    int xdma_arguments = 0;
    int qdma_arguments = 0;
    int new_arguments = 0;
    dayan_server_t *server = 0;
    struct sigaction action;
    int index;
    int status;

    (void)setvbuf(stderr, 0, _IONBF, 0);
    dayan_server_options_init(&options);
    (void)npu_hal_xilinx_dma_options_init(
        &dma, NPU_HAL_XILINX_DMA_PROFILE_SIMULATOR);
    options.hal.backend_options = &dma.base;
    options.firmware_path = DAYAN_DEFAULT_FIRMWARE;
    options.api_elf_path = DAYAN_DEFAULT_API_ELF;
    /* Resolve profile defaults before explicit overrides, independent of order. */
    for (index = 1; index < argc; index += 2) {
        npu_hal_xilinx_dma_profile_t profile;
        if (strcmp(argv[index], "--help") == 0) {
            usage(stdout, argv[0]);
            return 0;
        }
        if (index + 1 >= argc) {
            fprintf(stderr, "dayan_server: missing option value\n");
            return 2;
        }
        if (strcmp(argv[index], "--dma-profile") != 0 &&
            strcmp(argv[index], "--xdma-profile") != 0) continue;
        if (parse_dma_profile(argv[index + 1], &profile) < 0 ||
            npu_hal_xilinx_dma_options_init(&dma, profile) < 0) {
            fprintf(stderr, "dayan_server: invalid DMA profile\n");
            return 2;
        }
    }
    for (index = 1; index < argc; ++index) {
        const char *argument = argv[index];
        const char *value;

        if (strcmp(argument, "--help") == 0) {
            usage(stdout, argv[0]);
            return 0;
        }
        if (index + 1 >= argc) {
            fprintf(stderr, "dayan_server: missing value after %s\n",
                    argument);
            return 2;
        }
        value = argv[++index];
        status = 0;
        if (strncmp(argument, "--xdma-", 7) == 0 ||
            strcmp(argument, "--device-prefix") == 0 ||
            strcmp(argument, "--device-index") == 0 ||
            strcmp(argument, "--cluster-stride") == 0)
            xdma_arguments = 1;
        if (strncmp(argument, "--dma-", 6) == 0 || strcmp(argument, "--backend") == 0)
            new_arguments = 1;
        if (strcmp(argument, "--socket") == 0)
            options.socket_path = value;
        else if (strcmp(argument, "--firmware") == 0)
            options.firmware_path = value;
        else if (strcmp(argument, "--api-elf") == 0)
            options.api_elf_path = value;
        else if (strcmp(argument, "--backend") == 0) {
            if (strcmp(value, "xilinx-dma") != 0)
                status = -EINVAL;
        }
        else if (strcmp(argument, "--dma-driver") == 0) {
            if (strcmp(value, "xdma") == 0) dma.driver = NPU_HAL_XILINX_DMA_DRIVER_XDMA;
            else if (strcmp(value, "qdma") == 0) dma.driver = NPU_HAL_XILINX_DMA_DRIVER_QDMA;
            else status = -EINVAL;
        }
        else if (strcmp(argument, "--dma-completion") == 0) {
            if (strcmp(value, "irq") == 0) dma.completion = NPU_HAL_XILINX_DMA_COMPLETION_IRQ;
            else if (strcmp(value, "polling") == 0) dma.completion = NPU_HAL_XILINX_DMA_COMPLETION_POLLING;
            else status = -EINVAL;
        }
        else if (strcmp(argument, "--dma-device-name") == 0)
            dma.device_name = value;
        else if (strcmp(argument, "--dma-queue") == 0) {
            qdma_arguments = 1;
            status = parse_u32(value, &dma.queue);
        }
        else if (strcmp(argument, "--dma-user-irq") == 0) {
            qdma_arguments = 1;
            status = parse_u32(value, &dma.user_irq);
        }
        else if (strcmp(argument, "--dma-channel") == 0) {
            xdma_arguments = 1;
            status = parse_u32(value, &dma.channel);
        }
        else if (strcmp(argument, "--dma-event") == 0) {
            xdma_arguments = 1;
            status = parse_u32(value, &dma.event);
        }
        else if (strcmp(argument, "--xdma-profile") == 0 ||
                 strcmp(argument, "--dma-profile") == 0) {
            npu_hal_xilinx_dma_profile_t profile;
            status = parse_dma_profile(value, &profile);
        }
        else if (strcmp(argument, "--xdma-device-dir") == 0 ||
                 strcmp(argument, "--dma-device-dir") == 0 ||
                 strcmp(argument, "--device-prefix") == 0)
            dma.device_directory = value;
        else if (strcmp(argument, "--xdma-mmio-offset") == 0 ||
                 strcmp(argument, "--dma-mmio-offset") == 0)
            status = parse_u64(value,
                               &dma.mmio_address_offset);
        else if (strcmp(argument, "--device-index") == 0)
            status = parse_u32(value, &dma.device_index);
        else if (strcmp(argument, "--cluster-count") == 0)
            status = parse_u32(value, &options.hal.cluster_count);
        else if (strcmp(argument, "--cluster-stride") == 0)
            status = parse_u32(value,
                               &dma.cluster_device_stride);
        else if (strcmp(argument, "--cluster-mask") == 0)
            status = parse_u32(value, &options.enabled_cluster_mask);
        else if (strcmp(argument, "--core-mask") == 0)
            status = parse_u32(value, &options.enabled_core_mask);
        else if (strcmp(argument, "--timeout-ms") == 0) {
            status = parse_u32(value, &options.service.default_timeout_ms);
            if (status == 0 && !options.service.default_timeout_ms)
                status = -EINVAL;
        }
        else if (strcmp(argument, "--enable-mmu") == 0) {
            status = parse_u32(value, &options.service.enable_mmu);
            if (status == 0 && options.service.enable_mmu > 1U)
                status = -EINVAL;
        }
        else if (strcmp(argument, "--normal-memory-size-mib") == 0) {
            status = parse_u32(value,
                               &options.hal.normal_memory_size_mib);
            if (status == 0 && !options.hal.normal_memory_size_mib)
                status = -EINVAL;
        }
        else {
            fprintf(stderr, "dayan_server: unknown option %s\n", argument);
            return 2;
        }
        if (status < 0) {
            fprintf(stderr, "dayan_server: invalid value for %s: %s\n",
                    argument, value);
            return 2;
        }
    }
    if ((dma.driver == NPU_HAL_XILINX_DMA_DRIVER_QDMA && xdma_arguments) ||
        (dma.driver == NPU_HAL_XILINX_DMA_DRIVER_XDMA && qdma_arguments)) {
        fprintf(stderr, "dayan_server: options do not match selected DMA driver\n");
        return 2;
    }
    if (!new_arguments)
        dma.completion = NPU_HAL_XILINX_DMA_COMPLETION_LEGACY;
    fprintf(stderr, "dayan_server: DMA backend=xilinx-dma driver=%s profile=%s completion=%s\n",
            dma.driver == NPU_HAL_XILINX_DMA_DRIVER_QDMA ? "qdma" : "xdma",
            dma.profile == NPU_HAL_XILINX_DMA_PROFILE_FPGA ? "fpga" : "simulator",
            dma.completion == NPU_HAL_XILINX_DMA_COMPLETION_IRQ ? "irq" :
            dma.completion == NPU_HAL_XILINX_DMA_COMPLETION_POLLING ? "polling (1 us requested sleep)" : "legacy");
    memset(&action, 0, sizeof(action));
    action.sa_handler = stop_signal;
    sigemptyset(&action.sa_mask);
    (void)sigaction(SIGINT, &action, 0);
    (void)sigaction(SIGTERM, &action, 0);
    fprintf(stderr, "dayan_server: [main] starting (pid=%ld)\n",
            (long)getpid());
    status = dayan_server_create(&server, &options);
    if (status < 0) {
        fprintf(stderr,
                "dayan_server: [error] initialization failed: %s (%d)\n",
                strerror(-status), status);
        return 1;
    }
    fprintf(stderr,
            "dayan_server: [ready] accepting Runtime connections on %s\n",
            options.socket_path);
    status = dayan_server_run(server, (const volatile int *)&stop_requested);
    if (status < 0) {
        fprintf(stderr, "dayan_server: [error] service loop failed: %s (%d)\n",
                strerror(-status), status);
    } else if (stop_requested) {
        fprintf(stderr,
                "dayan_server: [shutdown] stop signal received; "
                "releasing resources\n");
    } else {
        fprintf(stderr,
                "dayan_server: [shutdown] service loop stopped; "
                "releasing resources\n");
    }
    dayan_server_destroy(server);
    if (status < 0) {
        fprintf(stderr,
                "dayan_server: [shutdown] stopped after service failure\n");
        return 1;
    }
    fprintf(stderr, "dayan_server: [shutdown] stopped cleanly\n");
    return 0;
}
