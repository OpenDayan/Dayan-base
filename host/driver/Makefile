CC ?= cc
CXX ?= c++
AR ?= ar
CLANG ?= clang
NM ?= nm

ROOT := ../..
BUILD := $(ROOT)/build/host/driver

CFLAGS ?= -O2 -g -std=c11 -Wall -Wextra -Werror -fPIC
CXXFLAGS ?= -O2 -g -std=c++17 -Wall -Wextra -Werror -fPIC
CPPFLAGS += -I$(ROOT)/host/include -I. -Ihal -Iserver -Iruntime \
	-iquote $(ROOT)/device/include

PROTOCOL_SOURCE := protocol.c
HAL_CONTROL_SOURCES := hal/hal.c hal/memory.c hal/fuse.c
HAL_CLIENT_SOURCES := hal/client.c hal/fuse_client.c
SERVICE_SOURCES := server/device.c server/elf.c server/rpc.c \
	server/kernel.c server/package.c server/profile.c
SERVER_SOURCES := server/scheduler.c server/server.c
RUNTIME_SOURCES := runtime/client.c runtime/objects.c runtime/transfer.c \
	runtime/kernel.c runtime/dayan_kernel.c runtime/package.c runtime/profile.c

PROTOCOL_OBJECT := $(BUILD)/protocol.o
HAL_CONTROL_OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(HAL_CONTROL_SOURCES))
HAL_CLIENT_OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(HAL_CLIENT_SOURCES))
SERVICE_OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(SERVICE_SOURCES))
SERVER_OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(SERVER_SOURCES))
RUNTIME_OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(RUNTIME_SOURCES))
ALL_OBJECTS := $(PROTOCOL_OBJECT) $(HAL_CONTROL_OBJECTS) \
	$(HAL_CLIENT_OBJECTS) $(SERVICE_OBJECTS) $(SERVER_OBJECTS) \
	$(RUNTIME_OBJECTS)
DEPS := $(ALL_OBJECTS:.o=.d) $(BUILD)/server/main.d

HAL_LIB := $(BUILD)/libDayanHal.a
SERVICE_LIB := $(BUILD)/libDayanService.a
RUNTIME_STATIC := $(BUILD)/libDayanRuntime.a
RUNTIME_SHARED := $(BUILD)/libDayanRuntime.so
RUNTIME_EXPORT_MAP := runtime/exports.map
COMPAT_LIB := $(BUILD)/libnpu_runtime.a
SERVER_BIN := $(BUILD)/dayan_server

.PHONY: all check clean cmodel-smoke cmodel-smoke-bin layer-check \
	runtime-export-check

all: $(HAL_LIB) $(SERVICE_LIB) $(RUNTIME_STATIC) $(RUNTIME_SHARED) \
	$(COMPAT_LIB) $(SERVER_BIN)

$(HAL_LIB): $(HAL_CONTROL_OBJECTS) $(HAL_CLIENT_OBJECTS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(SERVICE_LIB): $(PROTOCOL_OBJECT) $(SERVICE_OBJECTS) $(SERVER_OBJECTS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(RUNTIME_STATIC): $(PROTOCOL_OBJECT) $(HAL_CLIENT_OBJECTS) $(RUNTIME_OBJECTS)
	@mkdir -p $(dir $@)
	$(AR) rcs $@ $^

$(COMPAT_LIB): $(RUNTIME_STATIC)
	@mkdir -p $(dir $@)
	cp $< $@

$(RUNTIME_SHARED): $(PROTOCOL_OBJECT) $(HAL_CLIENT_OBJECTS) $(RUNTIME_OBJECTS) \
		$(RUNTIME_EXPORT_MAP)
	@mkdir -p $(dir $@)
	$(CC) -shared -pthread -Wl,-soname,libDayanRuntime.so \
		-Wl,--version-script,$(RUNTIME_EXPORT_MAP) -o $@ \
		$(PROTOCOL_OBJECT) $(HAL_CLIENT_OBJECTS) $(RUNTIME_OBJECTS)

$(SERVER_BIN): $(BUILD)/server/main.o $(PROTOCOL_OBJECT) \
		$(HAL_CONTROL_OBJECTS) $(SERVICE_OBJECTS) $(SERVER_OBJECTS)
	@mkdir -p $(dir $@)
	$(CC) -pthread -o $@ $^

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread -MMD -MP -c $< -o $@

$(BUILD)/tests/hal_test: hal/tests/hal_test.c $(HAL_CONTROL_OBJECTS) \
		$(HAL_CLIENT_OBJECTS)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread $^ -o $@

$(BUILD)/tests/package_test: runtime/tests/package_test.c \
		$(BUILD)/runtime/package.o $(BUILD)/server/package.o
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread $^ -o $@

$(BUILD)/tests/socket_test: runtime/tests/socket_test.c $(PROTOCOL_OBJECT) \
		$(HAL_CONTROL_OBJECTS) $(HAL_CLIENT_OBJECTS) $(SERVICE_OBJECTS) \
		$(SERVER_OBJECTS) $(RUNTIME_OBJECTS)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread $^ -o $@

$(BUILD)/tests/public_headers.o: runtime/tests/public_headers.cpp
	@mkdir -p $(dir $@)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@

runtime-export-check: $(RUNTIME_SHARED)
	@! $(NM) -D --defined-only $(RUNTIME_SHARED) | \
		awk 'NF >= 3 { name = $$3; sub(/@@.*/, "", name); print name }' | \
		grep -E '^(dayan_|runtime_|npu_(hal|ipc)_)'
	@$(NM) -D --defined-only $(RUNTIME_SHARED) | grep -q ' npu_runtime_open@@'
	@$(NM) -D --defined-only $(RUNTIME_SHARED) | \
		grep -q ' npu_kernel_descriptor_launch@@'

layer-check: $(RUNTIME_STATIC)
	@! rg -n '#include "(hal/hal\.h|server/)' protocol.c protocol.h runtime \
		--glob '!runtime/tests/**'
	@! rg -n '\bnpu_hal_(open|close|get_info|register_|irq_|boot_|memory_)' \
		runtime --glob '!runtime/tests/**'
	@! rg -n '\bdayan_[A-Za-z0-9_]+[[:space:]]*\(' $(ROOT)/host/include
	@! $(NM) -u $(RUNTIME_STATIC) | \
		grep -E ' (dayan_server_|npu_service_|npu_hal_(open|close|get_info|register_|irq_|boot_|memory_))'

check: all layer-check runtime-export-check $(BUILD)/tests/hal_test $(BUILD)/tests/package_test \
	$(BUILD)/tests/socket_test $(BUILD)/tests/public_headers.o
	$(BUILD)/tests/hal_test
	$(BUILD)/tests/package_test
	$(BUILD)/tests/socket_test

$(BUILD)/tests/cmodel_kernel.o: runtime/tests/cmodel_kernel.c
	@mkdir -p $(dir $@)
	$(CLANG) --target=riscv64-unknown-elf \
		-march=rv64imaf_zicsr_zifencei -mabi=lp64 -mcmodel=medany \
		-mno-relax -msmall-data-limit=0 -ffreestanding -fno-builtin \
		-fno-common -fno-pic -fno-stack-protector -ffunction-sections \
		-fdata-sections -O2 -Wall -Wextra -Werror \
		-I$(ROOT)/device/include/api -c $< -o $@

$(BUILD)/tests/cmodel_smoke: runtime/tests/cmodel_smoke.c \
		$(PROTOCOL_OBJECT) $(HAL_CONTROL_OBJECTS) $(HAL_CLIENT_OBJECTS) \
		$(SERVICE_OBJECTS) $(SERVER_OBJECTS) $(RUNTIME_OBJECTS)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread $^ -o $@

cmodel-smoke-bin: all $(BUILD)/tests/cmodel_kernel.o \
	$(BUILD)/tests/cmodel_smoke

cmodel-smoke: cmodel-smoke-bin
	sh ./run_cmodel_smoke.sh

clean:
	rm -rf $(BUILD)

-include $(DEPS)
