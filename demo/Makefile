ROOT := ..
BUILD := $(ROOT)/build/demo
DYCC := $(ROOT)/host/dycc/dycc
DYCC_BIN := $(ROOT)/build/host/dycc/dycc-bin
RUNTIME_LIB := $(ROOT)/build/host/driver/libDayanRuntime.a
NPU_KERNELS_DIR ?= $(abspath $(ROOT)/../npu-kernels)
NPU_KERNEL_INCLUDE := $(NPU_KERNELS_DIR)/include
MATMUL_SLICES ?= 3

PRINT_TARGET := $(BUILD)/print_demo
MATMUL_TARGET := $(BUILD)/matmul_bf16_1c
TARGETS := $(PRINT_TARGET) $(MATMUL_TARGET)
MATMUL_HEADERS := \
	$(NPU_KERNEL_INCLUDE)/npu_instruction.hpp \
	$(NPU_KERNEL_INCLUDE)/gen_npu_instruction_32.h \
	$(NPU_KERNEL_INCLUDE)/gen_npu_instruction_64.h \
	$(NPU_KERNEL_INCLUDE)/npu_lsu_csr.hpp \
	$(NPU_KERNEL_INCLUDE)/npu_mat_csr.hpp \
	device_include/cstdint

.PHONY: all check run run-matmul clean toolchain

all: $(TARGETS)

toolchain:
	$(MAKE) -C $(ROOT) host-driver dycc

$(DYCC_BIN) $(RUNTIME_LIB): | toolchain

$(PRINT_TARGET): print_demo.cpp $(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) -O2 -Wall -Wextra -Werror $< -o $@

$(MATMUL_TARGET): matmul_bf16_1c.cpp $(MATMUL_HEADERS) \
		$(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) -O2 -Wall -Wextra -Werror \
		-I$(NPU_KERNEL_INCLUDE) \
		-DMATMUL_BF16_1C_SLICES=$(MATMUL_SLICES) \
		-Xdevice -I$(abspath device_include) $< -o $@

check: all
	$(PRINT_TARGET) --help >/dev/null
	$(MATMUL_TARGET) --help >/dev/null

run: all
	bash ./run_cmodel.sh

run-matmul: all
	bash ./run_matmul_cmodel.sh

clean:
	rm -rf $(BUILD)
