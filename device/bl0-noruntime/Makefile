GCC=riscv64-unknown-elf-gcc
OBJCOPY=riscv64-unknown-elf-objcopy
OBJDUMP=riscv64-unknown-elf-objdump

.PHONY: clean all
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
	export QEMU="/workspace/npu-qemu/build/qemu-system-riscv64"
	export QEMU_FLAGS="-bios /workspace/npu-firmware/device/bl0/bootrom.elf -machine rvtng-v0 -nographic -m 4096"
	setsid ./run_bg.sh $$$$ &
	exec gdb-multiarch -q /workspace/npu-kernels/general_entry_fpga.elf -ex "target remote localhost:1234" -ex "b *0x80000000"
