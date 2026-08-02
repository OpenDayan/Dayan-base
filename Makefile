CLANG       ?= clang
OBJCOPY     ?= llvm-objcopy
OBJDUMP     ?= llvm-objdump
NM          ?= llvm-nm
READELF     ?= llvm-readelf
HOST_CC     ?= cc

BUILD       := build
NAME        := npu-os
ELF         := $(BUILD)/$(NAME).elf
BIN         := $(BUILD)/$(NAME).bin
DUMP        := $(BUILD)/$(NAME).objdump
MAP         := $(BUILD)/$(NAME).map

HARTS           ?= 4
CPU_LOCAL_SIZE  ?= 0x1000
STACK_SIZE      ?= 0x10000
HEAP_SIZE       ?= 0x100000
RAM_BASE        ?= 0x80000000
RAM_SIZE        ?= 0x04000000
COHERENT_BASE   ?= 0x90000000
COHERENT_SIZE   ?= 0x00400000

ARCH := rv64imaf_zicsr_zifencei
ABI  := lp64

CPPFLAGS := -Iinclude -Ilib/include \
            -DCONFIG_HART_COUNT=$(HARTS) \
            -DCONFIG_CPU_LOCAL_SIZE=$(CPU_LOCAL_SIZE) \
            -DCONFIG_STACK_SIZE=$(STACK_SIZE) \
            -DCONFIG_HEAP_SIZE=$(HEAP_SIZE)
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
           -Wl,--defsym=CONFIG_HEAP_SIZE=$(HEAP_SIZE) \
           -Wl,--defsym=CONFIG_RAM_BASE=$(RAM_BASE) \
           -Wl,--defsym=CONFIG_RAM_SIZE=$(RAM_SIZE) \
           -Wl,--defsym=CONFIG_COHERENT_BASE=$(COHERENT_BASE) \
           -Wl,--defsym=CONFIG_COHERENT_SIZE=$(COHERENT_SIZE)

C_SOURCES := $(wildcard boot/*.c lib/*.c mem/*.c platform/*.c rpc/*.c sche/*.c)
S_SOURCES := $(wildcard bl1/*.S sche/*.S)
OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(C_SOURCES)) \
           $(patsubst %.S,$(BUILD)/%.o,$(S_SOURCES))
DEPS := $(OBJECTS:.o=.d)
CONFIG_STAMP := $(BUILD)/.config
HOST_CFLAGS := -O2 -g -std=c11 -Wall -Wextra -Werror
HOST_TESTS := $(BUILD)/tests/list_test $(BUILD)/tests/rbtree_test \
              $(BUILD)/tests/mem_test $(BUILD)/tests/math_test

.PHONY: all clean check FORCE

all: $(ELF) $(BIN) $(DUMP)

$(ELF): $(OBJECTS) linker.ld $(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	$(CLANG) $(LDFLAGS) $(OBJECTS) -o $@

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

check: $(ELF) $(HOST_TESTS)
	@test -z "$$($(NM) --undefined-only $(ELF))"
	@if $(OBJDUMP) --disassemble --no-show-raw-insn $(ELF) | \
		grep -Eq '(^|[[:space:],(])gp([[:space:],)]|$$)'; then \
		echo "error: firmware must not access gp"; exit 1; \
	fi
	@if $(READELF) --relocations $(OBJECTS) | grep -q 'R_RISCV_GPREL'; then \
		echo "error: firmware must not contain gp-relative relocations"; exit 1; \
	fi
	@if $(NM) $(ELF) | grep -Fq '__global_pointer$$'; then \
		echo "error: firmware must not define __global_pointer$$"; exit 1; \
	fi
	@$(foreach test,$(HOST_TESTS),$(test) && ) true

$(BUILD)/tests/list_test: tests/list_test.c include/list.h include/container_of.h
	@mkdir -p $(dir $@)
	$(HOST_CC) $(HOST_CFLAGS) -Iinclude tests/list_test.c -o $@

$(BUILD)/tests/rbtree_test: tests/rbtree_test.c lib/rbtree.c \
                             include/rbtree.h include/container_of.h
	@mkdir -p $(dir $@)
	$(HOST_CC) $(HOST_CFLAGS) -Iinclude tests/rbtree_test.c lib/rbtree.c -o $@

$(BUILD)/tests/mem_test: tests/mem_test.c tests/include/cpu.h \
                         mem/mem.c lib/rbtree.c include/list.h \
                         include/rbtree.h include/container_of.h
	@mkdir -p $(dir $@)
	$(HOST_CC) $(HOST_CFLAGS) -Itests/include -Iinclude \
		tests/mem_test.c mem/mem.c lib/rbtree.c -o $@

$(BUILD)/tests/math_test: tests/math_test.c lib/math.c
	@mkdir -p $(dir $@)
	$(HOST_CC) $(HOST_CFLAGS) -fno-builtin tests/math_test.c lib/math.c -lm -o $@

$(CONFIG_STAMP): FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' '$(CLANG)' '$(CPPFLAGS)' '$(CFLAGS)' '$(LDFLAGS)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

FORCE:

clean:
	rm -rf $(BUILD)

-include $(DEPS)
