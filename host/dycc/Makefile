CXX := clang++
LLVM_CONFIG ?= llvm-config

ROOT := ../..
BUILD := $(ROOT)/build/host/dycc
TARGET := $(BUILD)/dycc-bin
SOURCE := src/dycc.cpp
LLVM_LIBDIR := $(shell $(LLVM_CONFIG) --libdir)
LLVM_PREFIX := $(shell $(LLVM_CONFIG) --prefix)
LLVM_INCLUDEDIR := $(shell $(LLVM_CONFIG) --includedir)

CLANG_SRC_INC := $(abspath $(LLVM_PREFIX)/../clang/include)
CLANG_GEN_INC := $(LLVM_PREFIX)/tools/clang/include

CPPFLAGS += -DDYCC_SOURCE_ROOT='"$(abspath $(ROOT))"' \
	    $(if $(wildcard $(CLANG_SRC_INC)),-I$(CLANG_SRC_INC)) \
	    $(if $(wildcard $(CLANG_GEN_INC)),-I$(CLANG_GEN_INC)) \
	    -I$(LLVM_INCLUDEDIR)
CXXFLAGS += $(shell $(LLVM_CONFIG) --cxxflags) -std=c++17 -Wall -Wextra \
	-Wno-unused-parameter
LDLIBS += -L$(LLVM_LIBDIR) -Wl,-rpath,$(LLVM_LIBDIR) -lclang-cpp \
	$(shell $(LLVM_CONFIG) --libs --system-libs)

.PHONY: all check clean

all: $(TARGET)

$(TARGET): $(SOURCE) include/dayan_compiler.h
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $< -o $@ $(LDLIBS)

check: $(TARGET)
	$(MAKE) -C ../driver all
	DYCC_BIN=$(abspath $(TARGET)) ./tests/test_dycc.sh

clean:
	rm -rf $(BUILD)
