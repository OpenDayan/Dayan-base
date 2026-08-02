CXX ?= c++

ROOT := ..
BUILD := $(ROOT)/build/demo
DYCC := $(ROOT)/host/dycc/dycc
RUNTIME_LIB := $(ROOT)/build/host/driver/libDayanRuntime.a

TARGET := $(BUILD)/printf_grid_demo
KERNEL_OBJECT := $(BUILD)/printf_grid/kernel.o
HOST_OBJECT := $(BUILD)/printf_grid/main.o

CPPFLAGS += -I$(ROOT)/host/include
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -Werror

.PHONY: all check run clean toolchain

all: $(TARGET)

toolchain:
	$(MAKE) -C $(ROOT) host-driver dycc

$(KERNEL_OBJECT): printf_grid/kernel.cpp | toolchain
	@mkdir -p $(dir $@)
	$(DYCC) -O2 -Wall -Wextra -Werror -c $< -o $@

$(HOST_OBJECT): printf_grid/main.cpp | toolchain
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(KERNEL_OBJECT) $(HOST_OBJECT) $(RUNTIME_LIB)
	@mkdir -p $(dir $@)
	$(CXX) $(CXXFLAGS) $^ -pthread -o $@

$(RUNTIME_LIB): | toolchain

check: all
	$(TARGET) --help >/dev/null

run: all
	sh ./run_cmodel.sh

clean:
	rm -rf $(BUILD)
