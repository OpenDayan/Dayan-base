CC ?= cc
AR ?= ar
CLANG ?= clang

ROOT := ../..
BUILD := $(ROOT)/build/host/runtime
LIB := $(BUILD)/libnpu_runtime.a
DAYAN_STATIC_LIB := $(BUILD)/libDayanRuntime.a
DAYAN_SHARED_LIB := $(BUILD)/libDayanRuntime.so
SOURCES := npu_driver.c npu_memory.c npu_runtime.c npu_elf.c npu_rpc.c \
	npu_kernel.c
OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(SOURCES))
DEPS := $(OBJECTS:.o=.d)
KERNEL_TEST_API_OBJ := $(BUILD)/tests/kernel_api.o
KERNEL_TEST_API_ELF := $(BUILD)/tests/kernel_api.elf
KERNEL_TEST_DEVICE_OBJ := $(BUILD)/tests/kernel_device.o
CFLAGS ?= -O2 -g -std=c11 -Wall -Wextra -Werror -fPIC
CPPFLAGS += -I../include -I. -idirafter $(ROOT)/include

.PHONY: all check clean cmodel-smoke cmodel-smoke-bin

all: $(LIB) $(DAYAN_STATIC_LIB) $(DAYAN_SHARED_LIB)

$(LIB) $(DAYAN_STATIC_LIB): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(DAYAN_SHARED_LIB): $(OBJECTS)
	@mkdir -p $(dir $@)
	$(CC) -shared -pthread -Wl,-soname,libDayanRuntime.so -o $@ $^

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread -MMD -MP -c $< -o $@

$(BUILD)/memory_test: tests/memory_test.c npu_memory.c runtime_internal.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread $< npu_memory.c -o $@

$(BUILD)/driver_test: tests/driver_test.c npu_driver.c ../include/npu_driver.h
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread $< npu_driver.c -o $@

$(KERNEL_TEST_API_OBJ): tests/kernel_api.c
	@mkdir -p $(dir $@)
	$(CLANG) --target=riscv64-unknown-elf -march=rv64imaf_zicsr_zifencei \
		-mabi=lp64 -mcmodel=medany -mno-relax -msmall-data-limit=0 \
		-ffreestanding -fno-builtin -fno-pic -c $< -o $@

$(KERNEL_TEST_API_ELF): $(KERNEL_TEST_API_OBJ)
	$(CLANG) --target=riscv64-unknown-elf -march=rv64imaf_zicsr_zifencei \
		-mabi=lp64 -mcmodel=medany -mno-relax -nostdlib -static \
		-fuse-ld=lld -Wl,-Ttext=0x20000,-e,test_api,--build-id=none \
		$< -o $@

$(KERNEL_TEST_DEVICE_OBJ): tests/kernel_device.c
	@mkdir -p $(dir $@)
	$(CLANG) --target=riscv64-unknown-elf -march=rv64imaf_zicsr_zifencei \
		-mabi=lp64 -mcmodel=medany -mno-relax -msmall-data-limit=0 \
		-ffreestanding -fno-builtin -fno-common -fno-pic \
		-fno-stack-protector -fno-unwind-tables \
		-fno-asynchronous-unwind-tables -ffunction-sections \
		-fdata-sections -O2 -c $< -o $@

$(BUILD)/kernel_loader_test: tests/kernel_loader_test.c $(LIB) \
		$(KERNEL_TEST_API_ELF) $(KERNEL_TEST_DEVICE_OBJ)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -I../include -pthread $< $(LIB) -o $@

$(BUILD)/cmodel_smoke: tests/cmodel_smoke.c $(LIB)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread $< $(LIB) -o $@

cmodel-smoke-bin: $(BUILD)/cmodel_smoke

cmodel-smoke: cmodel-smoke-bin
	./run_cmodel_smoke.sh

check: $(LIB) $(BUILD)/memory_test $(BUILD)/driver_test \
	$(BUILD)/kernel_loader_test $(KERNEL_TEST_API_ELF) \
	$(KERNEL_TEST_DEVICE_OBJ)
	$(BUILD)/memory_test
	$(BUILD)/driver_test
	$(BUILD)/kernel_loader_test $(KERNEL_TEST_API_ELF) \
		$(KERNEL_TEST_DEVICE_OBJ)

clean:
	rm -rf $(BUILD)

-include $(DEPS)
