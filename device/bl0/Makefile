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

GCC=riscv64-unknown-elf-gcc
OBJCOPY=riscv64-unknown-elf-objcopy
OBJDUMP=riscv64-unknown-elf-objdump
QEMU?=$(abspath $(CURDIR)/../../deps/npu-qemu/build/qemu-system-riscv64)
DEBUG_ELF?=../../build/device/npu-os.elf

.PHONY: clean all testDbg
.ONESHELL:

all: bootrom.img bootrom.elf bootrom.objdump

%.objdump: %.elf
	$(OBJDUMP) $< -S >$@

%.img: %.bin
	dd if=$< of=$@ bs=128 conv=sync

%.bin: %.elf
	$(OBJCOPY) -O binary $< $@

%.elf: %.S linker.ld
	$(GCC) -march=rv64g -Tlinker.ld -g $< -nostdlib -static -Wl,--no-gc-sections -o $@

clean:
	rm -f *.elf *.img *.objdump *.bin

testDbg:
	test -f "$(DEBUG_ELF)"
	export QEMU="$(QEMU)"
	export QEMU_FLAGS="-bios $(CURDIR)/bootrom.elf -machine rvtng-v0 -nographic -m 4096"
	setsid ./run_bg.sh $$$$ &
	exec gdb-multiarch -q "$(DEBUG_ELF)" -ex "target remote localhost:1234" -ex "b *0x80000000"
