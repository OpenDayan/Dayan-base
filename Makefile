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

INSTALL     ?= install

PREFIX      ?= /usr/local
DESTDIR     ?=
BINDIR      ?= $(PREFIX)/bin
INCLUDEDIR  ?= $(PREFIX)/include
LIBDIR      ?= $(PREFIX)/lib
DATADIR     ?= $(PREFIX)/share
DAYAN_SDKDIR ?= $(LIBDIR)/dayan
DAYAN_FIRMWAREDIR ?= $(DAYAN_SDKDIR)/firmware
NPU_QEMU_DIR ?= deps/npu-qemu
NPU_QEMU_BUILD_DIR ?= $(NPU_QEMU_DIR)/build
NPU_QEMU_BIN ?= $(NPU_QEMU_BUILD_DIR)/qemu-system-riscv64
NPU_QEMU_DL_BIN ?= $(NPU_QEMU_BUILD_DIR)/dl-riscv64

.PHONY: all clean check install qemu qemu-deps device device-check device-bl0 device-tests docs docs-check \
	host-driver host-driver-check dycc dycc-check dyperf dyperf-check \
	demo demo-check

all: qemu device host-driver dycc dyperf demo

qemu: $(NPU_QEMU_BIN) $(NPU_QEMU_DL_BIN)

qemu-deps:
	@if [ ! -f "$(NPU_QEMU_DIR)/configure" ]; then \
		git submodule update --init $(NPU_QEMU_DIR); \
	fi

$(NPU_QEMU_BUILD_DIR)/Makefile: | qemu-deps
	mkdir -p $(NPU_QEMU_BUILD_DIR)
	cd $(NPU_QEMU_BUILD_DIR) && ../configure --target-list=riscv64-softmmu

$(NPU_QEMU_BIN): $(NPU_QEMU_BUILD_DIR)/Makefile
	$(MAKE) -C $(NPU_QEMU_BUILD_DIR)

$(NPU_QEMU_DL_BIN): $(NPU_QEMU_BIN)
	$(MAKE) -C $(NPU_QEMU_BUILD_DIR) dl-riscv64

device:
	$(MAKE) -C device all

device-tests:
	$(MAKE) -C device device-tests

device-bl0:
	$(MAKE) -C device/bl0 all

device-check: qemu
	$(MAKE) -C device check QEMU_SYSTEM_RISCV64="$(abspath $(NPU_QEMU_BIN))"

host-driver:
	$(MAKE) -C host/driver all

host-driver-check: host-driver
	$(MAKE) -C host/driver check

dycc: host-driver
	$(MAKE) -C host/dycc all

dycc-check: dycc
	$(MAKE) -C host/dycc check

dyperf: host-driver
	$(MAKE) -C host/dyperf all

dyperf-check: host-driver-check
	$(MAKE) -C host/dyperf check

demo: host-driver dycc
	$(MAKE) -C demo all

demo-check: demo
	$(MAKE) -C demo check

docs:
	$(MAKE) -C dayan_kernel_docs

docs-check:
	$(MAKE) -C dayan_kernel_docs check

check: device-check host-driver-check dycc-check dyperf-check demo-check

install: all
	PREFIX="$(PREFIX)" DESTDIR="$(DESTDIR)" BINDIR="$(BINDIR)" \
		INCLUDEDIR="$(INCLUDEDIR)" LIBDIR="$(LIBDIR)" DATADIR="$(DATADIR)" \
		DAYAN_SDKDIR="$(DAYAN_SDKDIR)" \
		DAYAN_FIRMWAREDIR="$(DAYAN_FIRMWAREDIR)" INSTALL="$(INSTALL)" \
		./scripts/install.sh

clean:
	rm -rf build
	$(MAKE) -C device/bl0 clean
