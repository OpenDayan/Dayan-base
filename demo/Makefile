ROOT := ..
BUILD := $(ROOT)/build/demo
DYCC := $(ROOT)/host/dycc/dycc
DYCC_BIN := $(ROOT)/build/host/dycc/dycc-bin
RUNTIME_LIB := $(ROOT)/build/host/driver/libDayanRuntime.a
DAYAN_API_HEADERS := $(wildcard $(ROOT)/device/include/api/*.h)
MATMUL_SLICES ?= 3
MATMUL_BROADCAST_SLICES ?= 4
DEVICE_ISA_CHECK := ./check_device_isa.sh

PRINT_TARGET := $(BUILD)/print_demo
ADD_FP32_MACRO_TARGET := $(BUILD)/add_fp32_macro
ADD_FP32_VEC_TARGET := $(BUILD)/add_fp32_vec
FLASH_ATTENTION_1C_TARGET := $(BUILD)/flash_attention_1c
CACHE_SEMANTICS_TARGET := $(BUILD)/cache_semantics
NORMALIZATION_PROBE_TARGET := $(BUILD)/normalization_probe
TASK_GRID_PROBE_TARGET := $(BUILD)/task_grid_probe
CONTEXT_THREAD_PROBE_TARGET := $(BUILD)/context_thread_probe
TENSOR_INTERRUPT_PROBE_TARGET := $(BUILD)/tensor_interrupt_probe
LSU_BANDWIDTH_TARGET := $(BUILD)/lsu_bandwidth
SCHEDULER_BENCH_TARGET := $(BUILD)/scheduler_yield_bench
SHARED_NOTIFY_TARGET := $(BUILD)/shared_notify_latency
MATMUL_1C_TARGET := $(BUILD)/matmul_bf16_1c
MATMUL_4C_TARGET := $(BUILD)/matmul_bf16_4c
MATMUL_4C_BROADCAST_TARGET := $(BUILD)/matmul_bf16_broadcast_4c
PRINT_TEMPS := $(BUILD)/print_demo-temps
ADD_FP32_MACRO_TEMPS := $(BUILD)/add_fp32_macro-temps
ADD_FP32_VEC_TEMPS := $(BUILD)/add_fp32_vec-temps
FLASH_ATTENTION_1C_TEMPS := $(BUILD)/flash_attention_1c-temps
CACHE_SEMANTICS_TEMPS := $(BUILD)/cache_semantics-temps
NORMALIZATION_PROBE_TEMPS := $(BUILD)/normalization_probe-temps
TASK_GRID_PROBE_TEMPS := $(BUILD)/task_grid_probe-temps
CONTEXT_THREAD_PROBE_TEMPS := $(BUILD)/context_thread_probe-temps
TENSOR_INTERRUPT_PROBE_TEMPS := $(BUILD)/tensor_interrupt_probe-temps
LSU_BANDWIDTH_TEMPS := $(BUILD)/lsu_bandwidth-temps
SCHEDULER_BENCH_TEMPS := $(BUILD)/scheduler_yield_bench-temps
SHARED_NOTIFY_TEMPS := $(BUILD)/shared_notify_latency-temps
MATMUL_1C_TEMPS := $(BUILD)/matmul_bf16_1c-temps
MATMUL_4C_TEMPS := $(BUILD)/matmul_bf16_4c-temps
MATMUL_4C_BROADCAST_TEMPS := $(BUILD)/matmul_bf16_broadcast_4c-temps
TARGETS := $(PRINT_TARGET) $(ADD_FP32_MACRO_TARGET) $(ADD_FP32_VEC_TARGET) \
	$(FLASH_ATTENTION_1C_TARGET) \
	$(CACHE_SEMANTICS_TARGET) $(NORMALIZATION_PROBE_TARGET) $(TASK_GRID_PROBE_TARGET) \
	$(CONTEXT_THREAD_PROBE_TARGET) \
	$(TENSOR_INTERRUPT_PROBE_TARGET) \
	$(LSU_BANDWIDTH_TARGET) \
	$(SCHEDULER_BENCH_TARGET) $(SHARED_NOTIFY_TARGET) \
	$(MATMUL_1C_TARGET) $(MATMUL_4C_TARGET) \
	$(MATMUL_4C_BROADCAST_TARGET)
.PHONY: all check run run-context run-interrupt run-matmul run-matmul-4c \
	clean toolchain

all: $(TARGETS)

toolchain:
	$(MAKE) -C $(ROOT) host-driver dycc

$(DYCC_BIN) $(RUNTIME_LIB): | toolchain

$(PRINT_TARGET): print_demo.cpp $(DAYAN_API_HEADERS) $(DEVICE_ISA_CHECK) \
		$(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(PRINT_TEMPS) -O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) $(PRINT_TEMPS)/print_demo.cpp.dycc.device.o

$(ADD_FP32_MACRO_TARGET): add_fp32_macro.cpp $(DAYAN_API_HEADERS) \
		$(DEVICE_ISA_CHECK) $(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(ADD_FP32_MACRO_TEMPS) \
		-O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(ADD_FP32_MACRO_TEMPS)/add_fp32_macro.cpp.dycc.device.o

$(ADD_FP32_VEC_TARGET): add_fp32_vec.cpp $(DAYAN_API_HEADERS) \
		$(DEVICE_ISA_CHECK) $(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(ADD_FP32_VEC_TEMPS) \
		-O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(ADD_FP32_VEC_TEMPS)/add_fp32_vec.cpp.dycc.device.o

$(FLASH_ATTENTION_1C_TARGET): flash_attention_1c.cpp \
		$(DAYAN_API_HEADERS) $(DEVICE_ISA_CHECK) \
		$(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(FLASH_ATTENTION_1C_TEMPS) \
		-O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(FLASH_ATTENTION_1C_TEMPS)/flash_attention_1c.cpp.dycc.device.o

$(CACHE_SEMANTICS_TARGET): cache_semantics.cpp $(DAYAN_API_HEADERS) \
		$(DEVICE_ISA_CHECK) $(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(CACHE_SEMANTICS_TEMPS) \
		-O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(CACHE_SEMANTICS_TEMPS)/cache_semantics.cpp.dycc.device.o

$(NORMALIZATION_PROBE_TARGET): normalization_probe.cpp $(DAYAN_API_HEADERS) \
		$(DEVICE_ISA_CHECK) $(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(NORMALIZATION_PROBE_TEMPS) \
		-O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(NORMALIZATION_PROBE_TEMPS)/normalization_probe.cpp.dycc.device.o

$(TASK_GRID_PROBE_TARGET): task_grid_probe.cpp $(DAYAN_API_HEADERS) \
		$(DEVICE_ISA_CHECK) $(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(TASK_GRID_PROBE_TEMPS) \
		-O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(TASK_GRID_PROBE_TEMPS)/task_grid_probe.cpp.dycc.device.o

$(CONTEXT_THREAD_PROBE_TARGET): context_thread_probe.cpp \
		$(DAYAN_API_HEADERS) $(DEVICE_ISA_CHECK) \
		$(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(CONTEXT_THREAD_PROBE_TEMPS) \
		-O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(CONTEXT_THREAD_PROBE_TEMPS)/context_thread_probe.cpp.dycc.device.o

$(TENSOR_INTERRUPT_PROBE_TARGET): tensor_interrupt_probe.cpp \
		$(DAYAN_API_HEADERS) $(DEVICE_ISA_CHECK) \
		$(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(TENSOR_INTERRUPT_PROBE_TEMPS) \
		-O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(TENSOR_INTERRUPT_PROBE_TEMPS)/tensor_interrupt_probe.cpp.dycc.device.o

$(LSU_BANDWIDTH_TARGET): lsu_bandwidth.cpp \
		$(DAYAN_API_HEADERS) $(DEVICE_ISA_CHECK) \
		$(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(LSU_BANDWIDTH_TEMPS) -O2 -Wall -Wextra -Werror \
		$< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(LSU_BANDWIDTH_TEMPS)/lsu_bandwidth.cpp.dycc.device.o

$(SCHEDULER_BENCH_TARGET): scheduler_yield_bench.cpp $(DAYAN_API_HEADERS) \
		$(DEVICE_ISA_CHECK) $(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(SCHEDULER_BENCH_TEMPS) \
		-O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(SCHEDULER_BENCH_TEMPS)/scheduler_yield_bench.cpp.dycc.device.o

$(SHARED_NOTIFY_TARGET): shared_notify_latency.cpp $(DAYAN_API_HEADERS) \
		$(DEVICE_ISA_CHECK) $(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(SHARED_NOTIFY_TEMPS) \
		-O2 -Wall -Wextra -Werror $< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(SHARED_NOTIFY_TEMPS)/shared_notify_latency.cpp.dycc.device.o

$(MATMUL_1C_TARGET): matmul_bf16_1c.cpp $(DAYAN_API_HEADERS) \
		$(DEVICE_ISA_CHECK) \
		$(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(MATMUL_1C_TEMPS) -O2 -Wall -Wextra -Werror \
		-DMATMUL_BF16_1C_SLICES=$(MATMUL_SLICES) \
		$< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(MATMUL_1C_TEMPS)/matmul_bf16_1c.cpp.dycc.device.o

$(MATMUL_4C_TARGET): matmul_bf16_4c.cpp $(DAYAN_API_HEADERS) \
		$(DEVICE_ISA_CHECK) \
		$(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(MATMUL_4C_TEMPS) -O2 -Wall -Wextra -Werror \
		$< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(MATMUL_4C_TEMPS)/matmul_bf16_4c.cpp.dycc.device.o
$(MATMUL_4C_BROADCAST_TARGET): matmul_bf16_broadcast_4c.cpp \
		$(DAYAN_API_HEADERS) \
		$(DEVICE_ISA_CHECK) $(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) --save-temps=$(MATMUL_4C_BROADCAST_TEMPS) \
		-O2 -Wall -Wextra -Werror \
		-DMATMUL_BF16_4C_SLICES=$(MATMUL_BROADCAST_SLICES) \
		$< -o $@
	bash $(DEVICE_ISA_CHECK) \
		$(MATMUL_4C_BROADCAST_TEMPS)/matmul_bf16_broadcast_4c.cpp.dycc.device.o

check: all
	$(PRINT_TARGET) --help >/dev/null
	$(ADD_FP32_MACRO_TARGET) --help >/dev/null
	$(ADD_FP32_VEC_TARGET) --help >/dev/null
	$(FLASH_ATTENTION_1C_TARGET) --help >/dev/null
	$(CACHE_SEMANTICS_TARGET) --help >/dev/null
	$(NORMALIZATION_PROBE_TARGET) --help >/dev/null
	$(TASK_GRID_PROBE_TARGET) --help >/dev/null
	$(CONTEXT_THREAD_PROBE_TARGET) --help >/dev/null
	$(TENSOR_INTERRUPT_PROBE_TARGET) --help >/dev/null
	$(LSU_BANDWIDTH_TARGET) --help >/dev/null
	$(SCHEDULER_BENCH_TARGET) --help >/dev/null
	$(SHARED_NOTIFY_TARGET) --help >/dev/null
	$(MATMUL_1C_TARGET) --help >/dev/null
	$(MATMUL_4C_TARGET) --help >/dev/null
	$(MATMUL_4C_BROADCAST_TARGET) --help >/dev/null

run: all
	bash ./run_cmodel.sh

run-context: all
	DAYAN_DEMO_TEST=context bash ./run_cmodel.sh

run-interrupt: all
	DAYAN_DEMO_TEST=interrupt bash ./run_cmodel.sh

run-matmul: all
	bash ./run_matmul_cmodel.sh

run-matmul-4c: all
	bash ./run_matmul_4c_cmodel.sh

clean:
	rm -rf $(BUILD)
