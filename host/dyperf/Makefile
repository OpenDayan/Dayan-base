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
	python3 -c 'import json,sys; d=json.load(open(sys.argv[1])); \
	 e=d["traceEvents"]; assert len(e)==9; \
	 assert d["displayTimeUnit"]=="ns" and d["metadata"]["cycle_ns"]==1; \
	 assert e[1]["args"]["name"]=="worker"; \
	 assert e[2]["name"]=="WAIT_EVENT" and abs(e[2]["ts"]-4294967.280)<1e-9; \
	 assert e[3]["ph"]=="E" and abs(e[3]["ts"]-4294967.312)<1e-9; \
	 assert e[5]["args"]["name"]=="late\"child" and e[5]["tid"]==9; \
	 assert e[6]["name"]=="WAIT_START" and e[6]["ph"]=="B"' \
	 $(BUILD)/trace-test.json

clean:
	rm -rf $(BUILD)
