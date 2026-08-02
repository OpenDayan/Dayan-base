#define _POSIX_C_SOURCE 200809L

#include "server.h"

#include <errno.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

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
    if (errno || !*value || *end || parsed > UINT32_MAX)
        return -EINVAL;
    *result = (uint32_t)parsed;
    return 0;
}

static void usage(FILE *stream, const char *program)
{
    fprintf(stream,
            "usage: %s [options]\n"
            "  --socket PATH           Unix socket path\n"
            "  --firmware PATH         BL1 ELF\n"
            "  --api-elf PATH          public BL1 API ELF\n"
            "  --device-prefix PATH    FUSE/XDMA device directory\n"
            "  --device-index N        first XDMA device index\n"
            "  --cluster-count N       number of Clusters\n"
            "  --cluster-stride N      XDMA index stride per Cluster\n"
            "  --cluster-mask MASK     enabled Cluster mask\n"
            "  --core-mask MASK        enabled Core mask within Cluster\n"
            "  --help                  show this text\n",
            program);
}

int main(int argc, char **argv)
{
    dayan_server_options_t options;
    dayan_server_t *server = 0;
    struct sigaction action;
    int index;
    int status;

    dayan_server_options_init(&options);
    options.firmware_path = DAYAN_DEFAULT_FIRMWARE;
    options.api_elf_path = DAYAN_DEFAULT_API_ELF;
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
        if (strcmp(argument, "--socket") == 0)
            options.socket_path = value;
        else if (strcmp(argument, "--firmware") == 0)
            options.firmware_path = value;
        else if (strcmp(argument, "--api-elf") == 0)
            options.api_elf_path = value;
        else if (strcmp(argument, "--device-prefix") == 0)
            options.hal.device_prefix = value;
        else if (strcmp(argument, "--device-index") == 0)
            status = parse_u32(value, &options.hal.device_index);
        else if (strcmp(argument, "--cluster-count") == 0)
            status = parse_u32(value, &options.hal.cluster_count);
        else if (strcmp(argument, "--cluster-stride") == 0)
            status = parse_u32(value,
                               &options.hal.cluster_device_stride);
        else if (strcmp(argument, "--cluster-mask") == 0)
            status = parse_u32(value, &options.enabled_cluster_mask);
        else if (strcmp(argument, "--core-mask") == 0)
            status = parse_u32(value, &options.enabled_core_mask);
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
    memset(&action, 0, sizeof(action));
    action.sa_handler = stop_signal;
    sigemptyset(&action.sa_mask);
    (void)sigaction(SIGINT, &action, 0);
    (void)sigaction(SIGTERM, &action, 0);
    status = dayan_server_create(&server, &options);
    if (status < 0) {
        fprintf(stderr, "dayan_server: initialization failed: %d\n", status);
        return 1;
    }
    status = dayan_server_run(server, (const volatile int *)&stop_requested);
    dayan_server_destroy(server);
    if (status < 0) {
        fprintf(stderr, "dayan_server: service failed: %d\n", status);
        return 1;
    }
    return 0;
}
