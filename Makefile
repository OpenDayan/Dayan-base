CLANG       ?= clang
CLANGXX     ?= clang++
OBJCOPY     ?= llvm-objcopy
OBJDUMP     ?= llvm-objdump
NM          ?= llvm-nm
READELF     ?= llvm-readelf
HOST_CC     ?= cc

BUILD       := build
NAME        := npu-os
ELF         := $(BUILD)/$(NAME).elf
API_ELF     := $(BUILD)/$(NAME)-api.elf
BIN         := $(BUILD)/$(NAME).bin
DUMP        := $(BUILD)/$(NAME).objdump
MAP         := $(BUILD)/$(NAME).map
API_SYMBOLS := dayan-api.symbols
LOCAL_RESOURCE_APIS := dayan_resource_init,dayan_resource_alloc,dayan_resource_produce,dayan_resource_consume,dayan_resource_wait_ready,dayan_resource_wait_free
SHARED_RESOURCE_MUTATORS := dayan_shared_resource_alloc,dayan_shared_resource_produce,dayan_shared_resource_consume

HARTS           ?= 4
CPU_LOCAL_SIZE  ?= 0x1000
STACK_SIZE      ?= 0x10000
RPC_STACK_SIZE  ?= 0x1000
TRAP_STACK_SIZE ?= 0x1000
HEAP_SIZE       ?= 0x100000
RAM_BASE        ?= 0x80000000
RAM_SIZE        ?= 0x04000000
CONTROL_BASE    ?= 0x00000000
# Host-owned hardware window; these values are not passed to the linker.
COHERENT_BASE   ?= 0x00400000
COHERENT_SIZE   ?= 0x00400000

ARCH := rv64imaf_zicsr_zifencei
ABI  := lp64

CPPFLAGS := -Iinclude/api -Iinclude -Ilib/include \
            -DCONFIG_HART_COUNT=$(HARTS) \
            -DCONFIG_CPU_LOCAL_SIZE=$(CPU_LOCAL_SIZE) \
            -DCONFIG_STACK_SIZE=$(STACK_SIZE) \
            -DCONFIG_RPC_STACK_SIZE=$(RPC_STACK_SIZE) \
            -DCONFIG_TRAP_STACK_SIZE=$(TRAP_STACK_SIZE) \
            -DCONFIG_HEAP_SIZE=$(HEAP_SIZE) \
            -DCONFIG_CONTROL_BASE=$(CONTROL_BASE) \
            -DCONFIG_COHERENT_BASE=$(COHERENT_BASE) \
            -DCONFIG_COHERENT_SIZE=$(COHERENT_SIZE)
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
           -Wl,--defsym=CONFIG_RPC_STACK_SIZE=$(RPC_STACK_SIZE) \
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
HOST_CFLAGS := -O2 -g -std=c11 -Wall -Wextra -Werror
HOST_TESTS := $(BUILD)/tests/list_test $(BUILD)/tests/rbtree_test \
              $(BUILD)/tests/mem_test $(BUILD)/tests/math_test \
              $(BUILD)/tests/rpc_test
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

check: $(ELF) $(API_ELF) $(API_COMPILE_TEST) $(API_CXX_COMPILE_TEST) \
       $(HOST_TESTS)
	@test -z "$$($(NM) --undefined-only $(ELF))"
	@if $(READELF) --relocations $(OBJECTS) $(API_COMPILE_TEST) \
		$(API_CXX_COMPILE_TEST) | grep -q 'R_RISCV_GPREL'; then \
		echo "error: target objects must not contain gp-relative relocations"; \
		exit 1; \
	fi
	@if $(READELF) --relocations $(OBJECTS) $(API_COMPILE_TEST) \
		$(API_CXX_COMPILE_TEST) | grep -Eq 'R_RISCV_(TPREL|TLS_)'; then \
		echo "error: target objects must not use tp-relative TLS"; exit 1; \
	fi
	@if $(NM) $(ELF) | grep -Fq '__global_pointer$$'; then \
		echo "error: firmware must not define __global_pointer$$"; exit 1; \
	fi
	@if [ "$$($(OBJDUMP) --disassemble --no-show-raw-insn \
		$(BUILD)/sche/context.o | grep -Ec 'sd[[:space:]]+gp,')" -ne 1 ] || \
	   [ "$$($(OBJDUMP) --disassemble --no-show-raw-insn \
		$(BUILD)/sche/context.o | grep -Ec 'ld[[:space:]]+gp,')" -ne 2 ]; then \
		echo "error: context switch must save and restore per-proc gp"; exit 1; \
	fi
	@if ! $(OBJDUMP) --disassemble --no-show-raw-insn \
		$(API_COMPILE_TEST) $(API_CXX_COMPILE_TEST) | \
		grep -Eq '(^|[[:space:],(])gp([[:space:],)]|$$)'; then \
		echo "error: operator shared pointer must read gp"; exit 1; \
	fi
	@if ! $(OBJDUMP) --disassemble --no-show-raw-insn \
		$(BUILD)/bl1/bl1.o | \
		grep -Eq 'csrw[[:space:]]+mie, zero'; then \
		echo "error: BL1 must mask functional interrupts"; exit 1; \
	fi
	@if [ "$$($(OBJDUMP) --disassemble --no-show-raw-insn $(ELF) | \
		grep -Ec '[[:space:]](csrw|csrs|csrc|csrrw|csrrs|csrrc)[[:space:]].*mie')" \
		-ne 1 ]; then \
		echo "error: firmware must never reconfigure mie after BL1"; exit 1; \
	fi
	@if ! $(OBJDUMP) --disassemble --no-show-raw-insn \
		$(BUILD)/bl1/bl1.o | \
		grep -Eq 'csrc[[:space:]]+mstatus,.*' || \
	   ! $(OBJDUMP) --disassemble --no-show-raw-insn \
		$(BUILD)/bl1/bl1.o | \
		grep -Eq 'csrw[[:space:]]+mscratch,.*' || \
	   ! $(OBJDUMP) --disassemble --no-show-raw-insn \
		$(BUILD)/bl1/bl1.o | \
		grep -Eq 'csrw[[:space:]]+mtvec,.*'; then \
		echo "error: BL1 trap and interrupt state is incomplete"; exit 1; \
	fi
	@if ! $(OBJDUMP) --disassemble --no-show-raw-insn \
		$(BUILD)/platform/trap_entry.o | \
		grep -Eq 'csrrw[[:space:]]+sp, mscratch, sp'; then \
		echo "error: trap entry must switch to the per-hart trap stack"; \
		exit 1; \
	fi
	@for csr in mstatus mepc mcause mtval; do \
		if ! $(OBJDUMP) --disassemble --no-show-raw-insn \
			$(BUILD)/platform/trap_entry.o | \
			grep -Eq "csrr[[:space:]]+t0, $$csr"; then \
			echo "error: trap entry does not save $$csr"; exit 1; \
		fi; \
	done
	@if ! $(OBJDUMP) --disassemble --no-show-raw-insn \
		$(BUILD)/platform/trap_entry.o | grep -Eq 'frcsr[[:space:]]+t0'; then \
		echo "error: trap entry does not save fcsr"; exit 1; \
	fi
	@if [ "$$($(READELF) --sections --wide $(ELF) | \
		grep -Ec '\.(stack|rpc_stack|trap_stack)[[:space:]]+NOBITS')" \
		-ne 3 ]; then \
		echo "error: per-hart stacks must be NOBITS sections"; exit 1; \
	fi
	@stack_start=0x$$($(NM) --defined-only --format=posix $(ELF) | \
		awk '$$1 == "__stack_start" {print $$3}'); \
	 stack_end=0x$$($(NM) --defined-only --format=posix $(ELF) | \
		awk '$$1 == "__stack_end" {print $$3}'); \
	 rpc_start=0x$$($(NM) --defined-only --format=posix $(ELF) | \
		awk '$$1 == "__rpc_stack_start" {print $$3}'); \
	 rpc_end=0x$$($(NM) --defined-only --format=posix $(ELF) | \
		awk '$$1 == "__rpc_stack_end" {print $$3}'); \
	 trap_start=0x$$($(NM) --defined-only --format=posix $(ELF) | \
		awk '$$1 == "__trap_stack_start" {print $$3}'); \
	 trap_end=0x$$($(NM) --defined-only --format=posix $(ELF) | \
		awk '$$1 == "__trap_stack_end" {print $$3}'); \
	 if [ $$((stack_end - stack_start)) -ne \
	      $$(( $(HARTS) * $(STACK_SIZE) )) ] || \
	    [ $$((rpc_end - rpc_start)) -ne \
	      $$(( $(HARTS) * 2 * $(RPC_STACK_SIZE) )) ] || \
	    [ $$((trap_end - trap_start)) -ne \
	      $$(( $(HARTS) * $(TRAP_STACK_SIZE) )) ]; then \
		echo "error: linked per-hart stack reservation has the wrong size"; \
		exit 1; \
	 fi
	@for encoding in 0000220b 0000420b 0000620b 0000820b; do \
		if ! $(OBJDUMP) --disassemble $(BUILD)/sche/dapi.o | \
			grep -Fq "$$encoding"; then \
			echo "error: Produce instruction $$encoding is missing"; exit 1; \
		fi; \
	done
	@if ! $(OBJDUMP) --disassemble $(BUILD)/sche/dapi.o | \
		grep -Fq '0000150b'; then \
		echo "error: global Produce pop instruction encoding changed"; exit 1; \
	fi
	@if $(OBJDUMP) --no-show-raw-insn \
		--disassemble-symbols=$(LOCAL_RESOURCE_APIS) \
		$(BUILD)/sche/dapi.o | \
		grep -Eq '(^|[[:space:]])(amo[^[:space:]]*|lr\.[wd][^[:space:]]*|sc\.[wd][^[:space:]]*|fence)([[:space:]]|$$)'; then \
		echo "error: per-hart resource API must not use atomic instructions"; \
		exit 1; \
	fi
	@if ! $(OBJDUMP) --no-show-raw-insn \
		--disassemble-symbols=$(SHARED_RESOURCE_MUTATORS) \
		$(BUILD)/sche/dapi.o | grep -Fq 'amoswap.w.aq'; then \
		echo "error: shared resource API must use an atomic lock"; exit 1; \
	fi
	@LC_ALL=C sort -u $(API_SYMBOLS) > $(BUILD)/.api-symbols.expected
	@$(NM) --defined-only --extern-only --format=posix $(API_ELF) | \
		awk '{print $$1}' | LC_ALL=C sort -u > $(BUILD)/.api-symbols.actual
	@diff -u $(BUILD)/.api-symbols.expected $(BUILD)/.api-symbols.actual
	@$(OBJDUMP) --syms $(ELF) | \
		awk '$$4 == ".text.api.libc" || $$4 == ".text.api.dapi" \
		     {print $$NF}' | LC_ALL=C sort -u > $(BUILD)/.api-symbols.marked
	@diff -u $(BUILD)/.api-symbols.expected $(BUILD)/.api-symbols.marked
	@$(NM) --undefined-only --format=posix $(API_COMPILE_TEST) | \
		awk '{print $$1}' | LC_ALL=C sort -u > $(BUILD)/.api-symbols.used
	@comm -23 $(BUILD)/.api-symbols.used $(BUILD)/.api-symbols.expected > \
		$(BUILD)/.api-symbols.invalid
	@test ! -s $(BUILD)/.api-symbols.invalid || \
		{ echo "error: operator API test references non-public symbols"; \
		  cat $(BUILD)/.api-symbols.invalid; exit 1; }
	@$(foreach test,$(HOST_TESTS),$(test) && ) true

$(API_COMPILE_TEST): tests/api_compile.c $(wildcard include/api/*.h) $(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	$(CLANG) -Iinclude/api $(CFLAGS) -MMD -MP -c $< -o $@

$(API_CXX_COMPILE_TEST): tests/api_compile.c $(wildcard include/api/*.h) \
                         $(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	$(CLANGXX) -Iinclude/api $(filter-out -std=c11,$(CFLAGS)) \
		-std=c++17 -fno-exceptions -fno-rtti -x c++ -c $< -o $@

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

$(BUILD)/tests/rpc_test: tests/rpc_test.c include/rpc.h include/trap.h \
                         include/platform_regs.h
	@mkdir -p $(dir $@)
	$(HOST_CC) $(HOST_CFLAGS) -Iinclude tests/rpc_test.c -o $@

$(CONFIG_STAMP): FORCE
	@mkdir -p $(dir $@)
	@printf '%s\n' '$(CLANG)' '$(CLANGXX)' '$(CPPFLAGS)' '$(CFLAGS)' \
		'$(LDFLAGS)' > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; fi

FORCE:

clean:
	rm -rf $(BUILD)

-include $(DEPS)
