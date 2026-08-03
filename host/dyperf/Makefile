CC ?= cc

ROOT := ../..
BUILD := $(ROOT)/build/host/dyperf
TARGET := $(BUILD)/dyperf
JSON_TEST := $(BUILD)/json_test
SOURCE := dyperf.c
RUNTIME := $(ROOT)/build/host/driver/libDayanRuntime.a

CPPFLAGS += -I$(ROOT)/host/include
CFLAGS ?= -O2 -g -std=c11 -Wall -Wextra -Werror
LDLIBS += $(RUNTIME) -pthread

.PHONY: all check clean

all: $(TARGET)

$(RUNTIME):
	$(MAKE) -C ../driver all

$(TARGET): $(SOURCE) $(ROOT)/host/include/npu_runtime.h $(RUNTIME)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDLIBS)

$(JSON_TEST): tests/json_test.c $(SOURCE) \
		$(ROOT)/host/include/npu_runtime.h $(RUNTIME)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) $< -o $@ $(LDLIBS)

check: $(TARGET) $(JSON_TEST)
	$(TARGET) --help >/dev/null
	@! $(TARGET) --duration-ms invalid >/dev/null 2>&1
	@! $(TARGET) --follow --duration-ms 1 >/dev/null 2>&1
	$(JSON_TEST) $(BUILD)/trace-test.json
	python3 tests/validate_trace.py $(BUILD)/trace-test.json

clean:
	rm -rf $(BUILD)
