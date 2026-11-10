/***************************************************************************************
 * Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
 *
 * Dayan-base is licensed under the Mulan PSL v2.
 * You can use this software according to the terms and conditions of the Mulan PSL v2.
 * You may obtain a copy of Mulan PSL v2 at:
 *
 *     http://license.coscl.org.cn/MulanPSL2
 *
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
 * EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
 * MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
 *
 * See the Mulan PSL v2 for more details.
***************************************************************************************/

#include "hal_internal.h"
#include "platform_regs.h"

#include <errno.h>
#include <stdio.h>
#include <string.h>

/* Names emitted by npu-cluster's BuildConfigKey.  Match the whole config
 * before the '-' so unknown hardware never silently acquires a topology. */
static int parse_build_id(const char *id, uint32_t *cores, uint32_t *uram)
{
    static const char *const configs[] = {
        "VerifQualFullClusterConfig", "PerfSimQualFullClusterConfig",
        "VerifQualRtsClusterConfig", "QualFullClusterConfig",
        "VerifQualFullBlackBoxClusterConfig", "VerifSingleFullClusterConfig",
        "VerifSingleTinyClusterConfig", "PerfSimSingleFullClusterConfig",
        "SingleFullClusterConfig", "SingleTinyClusterConfig",
        "FpgaSingleSmallClusterConfig", "FpgaSingleFullClusterConfig",
        "FpgaSingleRtsClusterConfig", "FpgaSingleDmaClusterConfig",
        "FpgaSingleTinyClusterConfig", "FpgaQualFullClusterConfig",
        "FpgaQualRtsClusterConfig", "FpgaQualDmaClusterConfig",
    };
    const char *separator = strchr(id, '-');
    size_t length;

    if (!separator || !separator[1])
        return -EINVAL;
    length = (size_t)(separator - id);
    for (size_t i = 0; i < sizeof(configs) / sizeof(configs[0]); ++i) {
        if (strlen(configs[i]) != length || strncmp(id, configs[i], length))
            continue;
        *cores = strstr(configs[i], "Single") ? 1U : 4U;
        *uram = strstr(configs[i], "Tiny") ? DAYAN_URAM_SIZE_768K :
                                             DAYAN_URAM_SIZE_1536K;
        return 0;
    }
    return -ENOTSUP;
}

int npu_hal_detect_config(npu_hal_options_t *options,
                          npu_hal_build_id_reader_t read_id, void *state)
{
    uint32_t detected_cores = 0, detected_uram = 0;
    int complete = 1;
    int manual_cores = options->manual_cores_per_cluster ||
                       options->cores_per_cluster != 4U;
    int manual_uram = options->manual_uram_size ||
                      options->uram_size != DAYAN_URAM_SIZE;

    for (uint32_t cluster = 0; cluster < options->cluster_count; ++cluster) {
        char id[DAYAN_BOOTROM_BUILD_ID_SIZE];
        uint32_t cores, uram;
        int status = read_id(state, cluster, id, sizeof(id));
        int available = status >= 0 && memchr(id, 0, sizeof(id));

        fprintf(stderr, "dayan_hal: Cluster %u BootROM config=\"%s\"\n",
                cluster, available ? (*id ? id : "<empty>") : "<unavailable>");
        if (!available || parse_build_id(id, &cores, &uram) < 0) {
            fprintf(stderr, "dayan_hal: Warning: Cluster %u BootROM build "
                    "identifier unavailable or unrecognized; retaining "
                    "configured topology (cores=%u URAM=0x%x)\n", cluster,
                    options->cores_per_cluster, options->uram_size);
            complete = 0;
            continue;
        }
        fprintf(stderr, "dayan_hal: Cluster %u detected cores=%u "
                "URAM=0x%x\n", cluster, cores, uram);
        if (manual_cores && options->cores_per_cluster != cores)
            fprintf(stderr, "dayan_hal: Warning: Cluster %u manual "
                    "cores-per-cluster=%u differs from hardware=%u (%s)\n",
                    cluster, options->cores_per_cluster, cores, id);
        if (manual_uram && options->uram_size != uram)
            fprintf(stderr, "dayan_hal: Warning: Cluster %u manual "
                    "uram-size=0x%x differs from hardware=0x%x (%s)\n",
                    cluster, options->uram_size, uram, id);
        /* HAL currently exposes uniform cores and capacity across Clusters. */
        if (detected_cores &&
            ((!manual_cores && detected_cores != cores) ||
             (!manual_uram && detected_uram != uram))) {
            fprintf(stderr, "dayan_hal: incompatible Cluster hardware "
                    "configurations; uniform topology required\n");
            return -ENOTSUP;
        }
        detected_cores = cores;
        detected_uram = uram;
    }
    if (complete) {
        if (!manual_cores)
            options->cores_per_cluster = detected_cores;
        if (!manual_uram)
            options->uram_size = detected_uram;
    }
    return 0;
}
