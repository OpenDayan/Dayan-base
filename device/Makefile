CLANG       ?= clang
CLANGXX     ?= clang++
OBJCOPY     ?= llvm-objcopy
OBJDUMP     ?= llvm-objdump
NM          ?= llvm-nm
READELF     ?= llvm-readelf
QEMU_SYSTEM_RISCV64 ?=

ROOT        := ..
BUILD       ?= $(ROOT)/build/device
NAME        := npu-os
ELF         := $(BUILD)/$(NAME).elf
API_ELF     := $(BUILD)/$(NAME)-api.elf
BIN         := $(BUILD)/$(NAME).bin
DUMP        := $(BUILD)/$(NAME).objdump
MAP         := $(BUILD)/$(NAME).map
API_SYMBOLS := dayan-api.symbols

HARTS           ?= 4
CPU_LOCAL_SIZE  ?= 0x1000
STACK_SIZE      ?= 0x10000
PROC_SLOT_COUNT ?= 16
PROC_STACK_SIZE ?= 0x10000
PROC_PRINTF_SIZE ?= 0x1000
RPC_STACK_SIZE  ?= 0x1000
PROFILE_STACK_SIZE ?= 0x1000
TRAP_STACK_SIZE ?= 0x1000
HEAP_SIZE       ?= 0x100000
RAM_BASE        ?= 0x80000000
RAM_SIZE        ?= 0x04000000
CONTROL_BASE    ?= 0x00000000
# Host-owned hardware window; these values are not passed to the linker.
COHERENT_BASE   ?= 0x00400000
COHERENT_SIZE   ?= 0x00400000
URAM_BASE       ?= 0x00800000
URAM_SIZE       ?= 0x00180000
URAM_STRIDE     ?= 0x00200000

ARCH := rv64imaf_zicsr_zifencei_zacas
ABI  := lp64

CPPFLAGS := -Iinclude/api -Iinclude -Ilib/include \
            -DCONFIG_HART_COUNT=$(HARTS) \
            -DCONFIG_CPU_LOCAL_SIZE=$(CPU_LOCAL_SIZE) \
            -DCONFIG_STACK_SIZE=$(STACK_SIZE) \
            -DCONFIG_PROC_SLOT_COUNT=$(PROC_SLOT_COUNT) \
            -DCONFIG_PROC_STACK_SIZE=$(PROC_STACK_SIZE) \
            -DCONFIG_PROC_PRINTF_SIZE=$(PROC_PRINTF_SIZE) \
            -DCONFIG_RPC_STACK_SIZE=$(RPC_STACK_SIZE) \
            -DCONFIG_PROFILE_STACK_SIZE=$(PROFILE_STACK_SIZE) \
            -DCONFIG_TRAP_STACK_SIZE=$(TRAP_STACK_SIZE) \
            -DCONFIG_HEAP_SIZE=$(HEAP_SIZE) \
            -DCONFIG_CONTROL_BASE=$(CONTROL_BASE) \
            -DCONFIG_RAM_BASE=$(RAM_BASE) \
            -DCONFIG_RAM_SIZE=$(RAM_SIZE) \
            -DCONFIG_COHERENT_BASE=$(COHERENT_BASE) \
            -DCONFIG_COHERENT_SIZE=$(COHERENT_SIZE) \
            -DCONFIG_URAM_BASE=$(URAM_BASE) \
            -DCONFIG_URAM_SIZE=$(URAM_SIZE) \
            -DCONFIG_URAM_STRIDE=$(URAM_STRIDE)
CFLAGS := --target=riscv64-unknown-elf -march=$(ARCH) -mabi=$(ABI) \
          -mcmodel=medany -mno-relax -msmall-data-limit=0 \
          -O2 -g -std=c11 \
          -ffreestanding -fno-builtin -fno-common -fno-pic -fno-stack-protector \
          -fno-unwind-tables -fno-asynchronous-unwind-tables \
          -ffunction-sections -fdata-sections \
          -Wall -Wextra -Werror
LDFLAGS := --target=riscv64-unknown-elf -march=$(ARCH) -mabi=$(ABI) \
           -mcmodel=medany -mno-relax -nostdlib -static -fuse-ld=lld \
           -Wl,-T,linker.ld,-Map,$(MAP),--build-id=none \
           -Wl,--defsym=CONFIG_HART_COUNT=$(HARTS) \
           -Wl,--defsym=CONFIG_CPU_LOCAL_SIZE=$(CPU_LOCAL_SIZE) \
           -Wl,--defsym=CONFIG_STACK_SIZE=$(STACK_SIZE) \
           -Wl,--defsym=CONFIG_PROC_SLOT_COUNT=$(PROC_SLOT_COUNT) \
           -Wl,--defsym=CONFIG_PROC_STACK_SIZE=$(PROC_STACK_SIZE) \
           -Wl,--defsym=CONFIG_PROC_PRINTF_SIZE=$(PROC_PRINTF_SIZE) \
           -Wl,--defsym=CONFIG_RPC_STACK_SIZE=$(RPC_STACK_SIZE) \
           -Wl,--defsym=CONFIG_PROFILE_STACK_SIZE=$(PROFILE_STACK_SIZE) \
           -Wl,--defsym=CONFIG_TRAP_STACK_SIZE=$(TRAP_STACK_SIZE) \
           -Wl,--defsym=CONFIG_HEAP_SIZE=$(HEAP_SIZE) \
           -Wl,--defsym=CONFIG_RAM_BASE=$(RAM_BASE) \
           -Wl,--defsym=CONFIG_RAM_SIZE=$(RAM_SIZE)

C_SOURCES := $(wildcard boot/*.c lib/*.c mem/*.c platform/*.c rpc/*.c sche/*.c)
S_SOURCES := $(wildcard bl1/*.S platform/*.S sche/*.S)
OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(C_SOURCES)) \
           $(patsubst %.S,$(BUILD)/%.o,$(S_SOURCES))
DEPS := $(OBJECTS:.o=.d)
CONFIG_STAMP := $(BUILD)/.config
API_COMPILE_TEST := $(BUILD)/tests/api_compile.o
API_CXX_COMPILE_TEST := $(BUILD)/tests/api_compile.cxx.o

.PHONY: all clean check FORCE

all: $(ELF) $(API_ELF) $(BIN) $(DUMP)

$(ELF): $(OBJECTS) linker.ld $(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	$(CLANG) $(LDFLAGS) $(OBJECTS) -o $@

$(API_ELF): $(ELF) $(API_SYMBOLS)
	$(OBJCOPY) --strip-debug --strip-unneeded \
		--keep-symbols=$(API_SYMBOLS) $< $@

$(BIN): $(ELF)
	$(OBJCOPY) -O binary $< $@

$(DUMP): $(ELF)
	$(OBJDUMP) --section-headers --disassemble --source $< > $@

$(BUILD)/%.o: %.c $(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	$(CLANG) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

$(BUILD)/%.o: %.S $(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	$(CLANG) $(CPPFLAGS) $(CFLAGS) -MMD -MP -c $< -o $@

check: $(ELF) $(API_ELF) $(API_COMPILE_TEST) $(API_CXX_COMPILE_TEST)
	NM="$(NM)" OBJDUMP="$(OBJDUMP)" READELF="$(READELF)" \
		BUILD_DIR="$(abspath $(BUILD))" HARTS="$(HARTS)" \
		STACK_SIZE="$(STACK_SIZE)" RPC_STACK_SIZE="$(RPC_STACK_SIZE)" \
		PROFILE_STACK_SIZE="$(PROFILE_STACK_SIZE)" \
		PROC_SLOT_COUNT="$(PROC_SLOT_COUNT)" \
		PROC_STACK_SIZE="$(PROC_STACK_SIZE)" \
		PROC_PRINTF_SIZE="$(PROC_PRINTF_SIZE)" \
		TRAP_STACK_SIZE="$(TRAP_STACK_SIZE)" \
		./tests/check-firmware.sh
	CLANG="$(CLANG)" QEMU_SYSTEM_RISCV64="$(QEMU_SYSTEM_RISCV64)" \
		TEST_BUILD_DIR="$(abspath $(BUILD))/tests/qemu" \
		./tests/run-qemu-tests.sh

$(API_COMPILE_TEST): tests/api_compile.c $(wildcard include/api/*.h) \
		$(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	$(CLANG) -Iinclude/api $(CFLAGS) -MMD -MP -c $< -o $@

$(API_CXX_COMPILE_TEST): tests/api_compile.c $(wildcard include/api/*.h) \
		$(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	$(CLANGXX) -Iinclude/api $(filter-out -std=c11,$(CFLAGS)) \
		-std=c++17 -fno-exceptions -fno-rtti -x c++ -c $< -o $@

$(CONFIG_STAMP): FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' '$(CLANG)' '$(CLANGXX)' '$(CPPFLAGS)' '$(CFLAGS)' \
		'$(LDFLAGS)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

FORCE:

clean:
	rm -rf $(BUILD)

-include $(DEPS)
