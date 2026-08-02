ROOT := ..
BUILD := $(ROOT)/build/demo
DYCC := $(ROOT)/host/dycc/dycc
DYCC_BIN := $(ROOT)/build/host/dycc/dycc-bin
RUNTIME_LIB := $(ROOT)/build/host/driver/libDayanRuntime.a

TARGET := $(BUILD)/print_demo

.PHONY: all check run clean toolchain

all: $(TARGET)

toolchain:
	$(MAKE) -C $(ROOT) host-driver dycc

$(DYCC_BIN) $(RUNTIME_LIB): | toolchain

$(TARGET): print_demo.cpp $(DYCC_BIN) $(RUNTIME_LIB) | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) -O2 -Wall -Wextra -Werror $< -o $@

check: all
	$(TARGET) --help >/dev/null

run: all
	bash ./run_cmodel.sh

clean:
	rm -rf $(BUILD)
