ROOT := ..
BUILD := $(ROOT)/build/demo
DYCC := $(ROOT)/host/dycc/dycc
DYCC_BIN := $(ROOT)/build/host/dycc/dycc-bin
RUNTIME_LIB := $(ROOT)/build/host/driver/libDayanRuntime.a
NPU_KERNELS_DIR ?= $(abspath $(ROOT)/../npu-kernels)
NPU_KERNEL_INCLUDE := $(NPU_KERNELS_DIR)/include
DAYAN_API_HEADERS := $(wildcard $(ROOT)/device/include/api/*.h)
MATMUL_SLICES ?= 3
MATMUL_4C_TRACE ?= 0
DEVICE_ISA_CHECK := ./check_device_isa.sh

PRINT_TARGET := $(BUILD)/print_demo
MATMUL_1C_TARGET := $(BUILD)/matmul_bf16_1c
MATMUL_4C_TARGET := $(BUILD)/matmul_bf16_4c
PRINT_TEMPS := $(BUILD)/print_demo-temps
MATMUL_1C_TEMPS := $(BUILD)/matmul_bf16_1c-temps
MATMUL_4C_TEMPS := $(BUILD)/matmul_bf16_4c-temps
TARGETS := $(PRINT_TARGET) $(MATMUL_1C_TARGET) $(MATMUL_4C_TARGET)
MATMUL_HEADERS := \
	$(NPU_KERNEL_INCLUDE)/npu_instruction.hpp \
	$(NPU_KERNEL_INCLUDE)/gen_npu_instruction_32.h \
	$(NPU_KERNEL_INCLUDE)/gen_npu_instruction_64.h \
	$(NPU_KERNEL_INCLUDE)/npu_lsu_csr.hpp \
	$(NPU_KERNEL_INCLUDE)/npu_mat_csr.hpp \
	device_include/cstdint

.PHONY: all check run run-matmul run-matmul-4c clean toolchain

all: $(TARGETS)

toolchain:
	$(MAKE) -C $(ROOT) host-driver dycc

$(DYCC_BIN) $(RUNTIME_LIB): | toolchain

$(PRINT_TARGET): print_demo.cpp $(DAYAN_API_HEADERS) $(DEVICE_ISA_CHECK) \
		$(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(PRINT_TEMPS) -O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) $(PRINT_TEMPS)/print_demo.cpp.dycc.device.o

$(MATMUL_1C_TARGET): matmul_bf16_1c.cpp $(MATMUL_HEADERS) $(DAYAN_API_HEADERS) \
		$(DEVICE_ISA_CHECK) \
		$(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(MATMUL_1C_TEMPS) -O2 -Wall -Wextra -Werror \
		-I$(NPU_KERNEL_INCLUDE) \
		-DMATMUL_BF16_1C_SLICES=$(MATMUL_SLICES) \
		-Xdevice -I$(abspath device_include) $< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(MATMUL_1C_TEMPS)/matmul_bf16_1c.cpp.dycc.device.o

$(MATMUL_4C_TARGET): matmul_bf16_4c.cpp $(MATMUL_HEADERS) $(DAYAN_API_HEADERS) \
		$(DEVICE_ISA_CHECK) \
		$(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(MATMUL_4C_TEMPS) -O2 -Wall -Wextra -Werror \
		-I$(NPU_KERNEL_INCLUDE) \
		-DMATMUL_BF16_4C_SLICES=$(MATMUL_SLICES) \
		-DMATMUL_BF16_4C_TRACE=$(MATMUL_4C_TRACE) \
		-Xdevice -I$(abspath device_include) $< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(MATMUL_4C_TEMPS)/matmul_bf16_4c.cpp.dycc.device.o

check: all
	$(PRINT_TARGET) --help >/dev/null
	$(MATMUL_1C_TARGET) --help >/dev/null
	$(MATMUL_4C_TARGET) --help >/dev/null

run: all
	bash ./run_cmodel.sh

run-matmul: all
	bash ./run_matmul_cmodel.sh

run-matmul-4c: all
	bash ./run_matmul_4c_cmodel.sh

clean:
	rm -rf $(BUILD)
