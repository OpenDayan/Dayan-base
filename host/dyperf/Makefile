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

CC ?= cc

ROOT := ../..
BUILD := $(ROOT)/build/host/dyperf
TARGET := $(BUILD)/dyperf
JSON_TEST := $(BUILD)/json_test
PCU_TEST := $(BUILD)/pcu_test
SOURCE := dyperf.c
RUNTIME := $(ROOT)/build/host/driver/libDayanRuntime.a

CPPFLAGS += -I$(ROOT)/host/include
CFLAGS ?= -O2 -g -std=c11 -Wall -Wextra -Werror
LDLIBS += $(RUNTIME) -pthread -lm

.PHONY: all check clean

all: $(TARGET)

$(RUNTIME):
	$(MAKE) -C ../driver all

$(TARGET): $(SOURCE) $(ROOT)/host/include/npu_runtime.h \
		$(ROOT)/host/include/npu_pcu.h $(RUNTIME)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDLIBS)

$(JSON_TEST): tests/json_test.c $(SOURCE) \
		$(ROOT)/host/include/npu_runtime.h $(ROOT)/host/include/npu_pcu.h $(RUNTIME)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDLIBS)

$(PCU_TEST): tests/pcu_test.c $(SOURCE) \
		$(ROOT)/host/include/npu_runtime.h $(ROOT)/host/include/npu_pcu.h $(RUNTIME)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDLIBS)

check: $(TARGET) $(JSON_TEST) $(PCU_TEST)
	$(TARGET) --help >/dev/null
	@! $(TARGET) --duration-ms invalid >/dev/null 2>&1
	@! $(TARGET) --follow --duration-ms 1 >/dev/null 2>&1
	@! $(TARGET) --pcu-interval nan >/dev/null 2>&1
	@! $(TARGET) --pcu-interval -0.1 >/dev/null 2>&1
	@! $(TARGET) --clock-hz 0 >/dev/null 2>&1
	@! $(TARGET) --lsu-beat-bytes 3 >/dev/null 2>&1
	$(JSON_TEST) $(BUILD)/trace-test.json
	python3 tests/validate_trace.py $(BUILD)/trace-test.json
	$(PCU_TEST) $(BUILD)/pcu-test.json
	python3 tests/validate_pcu.py $(BUILD)/pcu-test.json

clean:
	rm -rf $(BUILD)
