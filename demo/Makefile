#***************************************************************************************
# Copyright (c) 2026 Institute of Computing Technology, Chinese Academy of Sciences
#
# Dayan-base is licensed under the Mulan PSL v2.
# You can use this software according to the terms and conditions of the Mulan PSL v2.
# You may obtain a copy of Mulan PSL v2 at:
#
#     http://license.coscl.org.cn/MulanPSL2
#
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND,
# EITHER EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT,
# MERCHANTABILITY OR FIT FOR A PARTICULAR PURPOSE.
#
# See the Mulan PSL v2 for more details.
#***************************************************************************************

ROOT := ..
CPPFLAGS += -I$(CURDIR)
BUILD ?= $(ROOT)/build/demo
DYCC := $(ROOT)/host/dycc/dycc
DYCC_BIN := $(ROOT)/build/host/dycc/dycc-bin
RUNTIME_LIB := $(ROOT)/build/host/driver/libDayanRuntime.a
DAYAN_API_HEADERS := $(wildcard $(ROOT)/device/include/api/*.h) \
	$(ROOT)/host/include/npu_runtime.h uram_config.h
DEVICE_ISA_CHECK := ./check_device_isa.sh

# One source per executable, without variant-specific compiler flags.
SIMPLE_DEMOS := print_demo add_fp32_macro add_fp32_vec \
	task_grid_probe context_thread_probe tensor_interrupt_probe \
	lsu_bandwidth embedding_dma_bandwidth vec_peak scheduler_yield_bench shared_notify_latency \
	launch_latency matmul_bf16_1c matmul_bf16_4c matmul_fp8_4c matmul_fp4_1c matmul_fp4_4c
SIMPLE_TARGETS := $(addprefix $(BUILD)/,$(SIMPLE_DEMOS))

FLASH_ATTENTION_DEMOS := flash_attention flash_attention_tgmt_barrier \
	flash_attention_tgmt_shared_4c flash_attention_tgmt_resource_4c \
	flash_attention_tgmt_resource_timed_4c flash_attention_tgmt_resource_opt_4c
FLASH_ATTENTION_TARGETS := $(addprefix $(BUILD)/,$(FLASH_ATTENTION_DEMOS))
FLASH_ATTENTION_TIMED_TARGETS := $(BUILD)/flash_attention_tgmt_resource_timed_4c \
	$(BUILD)/flash_attention_tgmt_resource_opt_4c
GDN_TARGETS := $(BUILD)/gdn_tgmt $(BUILD)/gdn_sw_vec_tgmt $(BUILD)/gdn_sw
TARGETS := $(SIMPLE_TARGETS) $(FLASH_ATTENTION_TARGETS) $(GDN_TARGETS)

.PHONY: all list check clean toolchain

all: $(TARGETS)

list:
	@printf '%s\n' $(notdir $(TARGETS))

toolchain:
	$(MAKE) -C $(ROOT) host-driver dycc

$(DYCC_BIN) $(RUNTIME_LIB): | toolchain

$(SIMPLE_TARGETS): $(BUILD)/%: %.cpp $(DAYAN_API_HEADERS) $(DEVICE_ISA_CHECK) \
		$(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) $(CPPFLAGS) --save-temps=$@-temps -O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) $@-temps/$*.cpp.dycc.device.o

$(FLASH_ATTENTION_TARGETS): flash_attention.cpp $(DAYAN_API_HEADERS) \
		$(DEVICE_ISA_CHECK) $(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) $(CPPFLAGS) --save-temps=$@-temps \
		-DFLASH_ATTENTION_BARRIER_ONLY=$(if $(filter %/flash_attention_tgmt_barrier %/flash_attention_tgmt_shared_4c,$@),1,0) \
		-DFLASH_ATTENTION_SHARED_KV=$(if $(filter %/flash_attention_tgmt_shared_4c,$@),1,0) \
		-DFLASH_ATTENTION_RESOURCE_4C=$(if $(filter %/flash_attention_tgmt_resource_4c %/flash_attention_tgmt_resource_timed_4c %/flash_attention_tgmt_resource_opt_4c,$@),1,0) \
		-DFLASH_ATTENTION_TIMING=$(if $(filter $(FLASH_ATTENTION_TIMED_TARGETS),$@),1,0) \
		-DFLASH_ATTENTION_OPT=$(if $(filter %/flash_attention_tgmt_resource_opt_4c,$@),1,0) \
		-O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) $@-temps/flash_attention.cpp.dycc.device.o

$(GDN_TARGETS): gdn.cpp gdn_chunk.h $(DAYAN_API_HEADERS) $(DEVICE_ISA_CHECK) \
		$(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) $(CPPFLAGS) --save-temps=$@-temps \
		-DGDN_SOFTWARE_PIPELINE=$(if $(filter %/gdn_tgmt,$@),0,1) \
		-DGDN_VEC_TGMT=$(if $(filter %/gdn_sw,$@),0,1) \
		-O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) $@-temps/gdn.cpp.dycc.device.o

# Host-only smoke checks; --help exits before opening a device connection.
check: all
	@set -e; for target in $(TARGETS); do "$$target" --help >/dev/null; done
	@echo "demo-check: PASS ($(words $(TARGETS)) executables)"

clean:
	rm -rf $(BUILD)
