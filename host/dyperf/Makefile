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
	$(JSON_TEST) $(BUILD)/trace-test.json
	python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); \
	 e=d["traceEvents"]; assert len(e)==6; \
	 assert e[2]["name"]=="WAIT_EVENT" and e[2]["ts"]==4294967280; \
	 assert e[3]["ph"]=="E" and e[3]["ts"]==4294967312; \
	 assert e[4]["name"]=="READY" and e[4]["ph"]=="B"' \
	 $(BUILD)/trace-test.json

clean:
	rm -rf $(BUILD)
