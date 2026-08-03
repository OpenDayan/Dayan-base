INSTALL     ?= install

PREFIX      ?= /usr/local
DESTDIR     ?=
BINDIR      ?= $(PREFIX)/bin
INCLUDEDIR  ?= $(PREFIX)/include
LIBDIR      ?= $(PREFIX)/lib
DATADIR     ?= $(PREFIX)/share
DAYAN_SDKDIR ?= $(LIBDIR)/dayan
DAYAN_FIRMWAREDIR ?= $(DAYAN_SDKDIR)/firmware

.PHONY: all clean check install device device-check docs docs-check \
	host-driver host-driver-check dycc dycc-check dyperf dyperf-check \
	demo demo-check

all: device host-driver dycc dyperf

device:
	$(MAKE) -C device all

device-check:
	$(MAKE) -C device check

host-driver:
	$(MAKE) -C host/driver all

host-driver-check:
	$(MAKE) -C host/driver check

dycc:
	$(MAKE) -C host/dycc all

dycc-check:
	$(MAKE) -C host/dycc check

dyperf: host-driver
	$(MAKE) -C host/dyperf all

dyperf-check: host-driver-check
	$(MAKE) -C host/dyperf check

demo:
	$(MAKE) -C demo all

demo-check:
	$(MAKE) -C demo check

docs:
	$(MAKE) -C dayan_kernel_docs

docs-check:
	$(MAKE) -C dayan_kernel_docs check

check: device-check host-driver-check dycc-check dyperf-check

install: all
	PREFIX="$(PREFIX)" DESTDIR="$(DESTDIR)" BINDIR="$(BINDIR)" \
		INCLUDEDIR="$(INCLUDEDIR)" LIBDIR="$(LIBDIR)" DATADIR="$(DATADIR)" \
		DAYAN_SDKDIR="$(DAYAN_SDKDIR)" \
		DAYAN_FIRMWAREDIR="$(DAYAN_FIRMWAREDIR)" INSTALL="$(INSTALL)" \
		./scripts/install.sh

clean:
	rm -rf build
