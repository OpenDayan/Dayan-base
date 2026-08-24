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
	-Ihal/include \
	-iquote $(ROOT)/device/include

PROTOCOL_SOURCE := protocol.c
HAL_ROOT_SOURCES := $(wildcard hal/*.c)
HAL_BACKEND_DIRS := $(wildcard hal/backend/*/)
HAL_BACKEND_SOURCES := $(foreach dir,$(HAL_BACKEND_DIRS),$(wildcard $(dir)*.c))
HAL_IO_SOURCES := $(filter %/io.c,$(HAL_BACKEND_SOURCES))
HAL_DATA_SOURCES := hal/dataplane.c \
	$(filter %/dataplane.c,$(HAL_BACKEND_SOURCES))
HAL_CONTROL_SOURCES := $(filter-out hal/dataplane.c,$(HAL_ROOT_SOURCES)) \
	$(filter-out %/dataplane.c %/io.c,$(HAL_BACKEND_SOURCES))
SERVICE_SOURCES := server/device.c server/elf.c server/rpc.c \
	server/kernel.c server/package.c server/profile.c
SERVER_SOURCES := server/scheduler.c server/server.c
RUNTIME_SOURCES := $(wildcard runtime/*.c)

PROTOCOL_OBJECT := $(BUILD)/protocol.o
HAL_IO_OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(HAL_IO_SOURCES))
HAL_CONTROL_OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(HAL_CONTROL_SOURCES))
HAL_DATA_OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(HAL_DATA_SOURCES))
SERVICE_OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(SERVICE_SOURCES))
SERVER_OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(SERVER_SOURCES))
RUNTIME_OBJECTS := $(patsubst %.c,$(BUILD)/%.o,$(RUNTIME_SOURCES))
ALL_OBJECTS := $(PROTOCOL_OBJECT) $(HAL_IO_OBJECTS) \
	$(HAL_CONTROL_OBJECTS) $(HAL_DATA_OBJECTS) $(SERVICE_OBJECTS) $(SERVER_OBJECTS) \
	$(RUNTIME_OBJECTS)
DEPS := $(ALL_OBJECTS:.o=.d) $(BUILD)/server/main.d

HAL_LIB := $(BUILD)/libDayanHal.a
SERVICE_LIB := $(BUILD)/libDayanService.a
RUNTIME_STATIC := $(BUILD)/libDayanRuntime.a
RUNTIME_SHARED := $(BUILD)/libDayanRuntime.so
RUNTIME_EXPORT_MAP := runtime/exports.map
COMPAT_LIB := $(BUILD)/libnpu_runtime.a
SERVER_BIN := $(BUILD)/dayan_server

.PHONY: all check clean cmodel-smoke cmodel-smoke-bin \
	consecutive-launch-test-bin layer-check runtime-export-check

all: $(HAL_LIB) $(SERVICE_LIB) $(RUNTIME_STATIC) $(RUNTIME_SHARED) \
	$(COMPAT_LIB) $(SERVER_BIN)

$(HAL_LIB): $(HAL_IO_OBJECTS) $(HAL_CONTROL_OBJECTS) $(HAL_DATA_OBJECTS)
	@mkdir -p $(dir $@)
	$(RM) $@
	$(AR) rcs $@ $^

$(SERVICE_LIB): $(PROTOCOL_OBJECT) $(SERVICE_OBJECTS) $(SERVER_OBJECTS)
	@mkdir -p $(dir $@)
	$(RM) $@
	$(AR) rcs $@ $^

$(RUNTIME_STATIC): $(PROTOCOL_OBJECT) $(HAL_IO_OBJECTS) \
		$(HAL_DATA_OBJECTS) $(RUNTIME_OBJECTS)
	@mkdir -p $(dir $@)
	$(RM) $@
	$(AR) rcs $@ $^

$(COMPAT_LIB): $(RUNTIME_STATIC)
	@mkdir -p $(dir $@)
	cp $< $@

$(RUNTIME_SHARED): $(PROTOCOL_OBJECT) $(HAL_IO_OBJECTS) \
		$(HAL_DATA_OBJECTS) $(RUNTIME_OBJECTS) $(RUNTIME_EXPORT_MAP)
	@mkdir -p $(dir $@)
	$(CC) -shared -pthread -Wl,-soname,libDayanRuntime.so \
		-Wl,--version-script,$(RUNTIME_EXPORT_MAP) -o $@ \
		$(PROTOCOL_OBJECT) $(HAL_IO_OBJECTS) $(HAL_DATA_OBJECTS) \
		$(RUNTIME_OBJECTS)

$(SERVER_BIN): $(BUILD)/server/main.o $(PROTOCOL_OBJECT) \
		$(HAL_IO_OBJECTS) $(HAL_CONTROL_OBJECTS) $(SERVICE_OBJECTS) \
		$(SERVER_OBJECTS)
	@mkdir -p $(dir $@)
	$(CC) -pthread -o $@ $^

$(BUILD)/%.o: %.c
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread -MMD -MP -c $< -o $@

$(BUILD)/tests/hal_test: hal/tests/hal_test.c $(HAL_IO_OBJECTS) \
		$(HAL_CONTROL_OBJECTS) $(HAL_DATA_OBJECTS)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread $^ -o $@

$(BUILD)/tests/package_test: runtime/tests/package_test.c \
		$(BUILD)/runtime/package.o $(BUILD)/server/package.o
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread $^ -o $@

$(BUILD)/tests/socket_test: runtime/tests/socket_test.c $(PROTOCOL_OBJECT) \
		$(HAL_IO_OBJECTS) $(HAL_CONTROL_OBJECTS) $(HAL_DATA_OBJECTS) \
		$(SERVICE_OBJECTS) \
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
	@$(NM) -D --defined-only $(RUNTIME_SHARED) | \
		grep -Eq ' npu_runtime_open(@@[^ ]+)?$$'
	@$(NM) -D --defined-only $(RUNTIME_SHARED) | \
		grep -Eq ' npu_kernel_descriptor_launch(@@[^ ]+)?$$'

layer-check: $(RUNTIME_STATIC)
	@! rg -n '#include "(hal\.h|server/)' protocol.c protocol.h runtime \
		--glob '!runtime/tests/**'
	@! rg -n '\bnpu_hal_xdma|\bNPU_HAL_XDMA_' \
		hal/config.c hal/include/npu_hal_config.h server/server.c server/server.h
	@if rg -n '#include <sys/socket\.h>|\b(socket|socketpair|sendmsg|recvmsg|sendto|recvfrom|connect|accept|bind|listen)[[:space:]]*\(' hal/backend; then \
		exit 1; \
	else \
		test $$? -eq 1; \
	fi
	@! rg -n '\bnpu_hal_(open|close|get_info|register_|irq_|boot_|memory_)' \
		runtime --glob '!runtime/tests/**'
	@! rg -n '\bdayan_[A-Za-z0-9_]+[[:space:]]*\(' $(ROOT)/host/include
	@! $(NM) -u $(RUNTIME_STATIC) | \
		grep -E ' (dayan_server_|npu_service_|npu_hal_(open|close|get_info|register_|irq_|boot_|memory_))'

check: all layer-check runtime-export-check $(BUILD)/tests/hal_test \
	$(BUILD)/tests/package_test $(BUILD)/tests/socket_test \
	$(BUILD)/tests/public_headers.o
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
		$(PROTOCOL_OBJECT) $(HAL_IO_OBJECTS) $(HAL_CONTROL_OBJECTS) \
		$(HAL_DATA_OBJECTS) \
		$(SERVICE_OBJECTS) $(SERVER_OBJECTS) $(RUNTIME_OBJECTS)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread $^ -o $@

$(BUILD)/tests/consecutive_launches: runtime/tests/consecutive_launches.c \
		$(PROTOCOL_OBJECT) $(HAL_IO_OBJECTS) $(HAL_DATA_OBJECTS) $(RUNTIME_OBJECTS)
	@mkdir -p $(dir $@)
	$(CC) $(CPPFLAGS) $(CFLAGS) -pthread $^ -o $@

cmodel-smoke-bin: all $(BUILD)/tests/cmodel_kernel.o \
	$(BUILD)/tests/cmodel_smoke $(BUILD)/tests/consecutive_launches

consecutive-launch-test-bin: all $(BUILD)/tests/cmodel_kernel.o \
	$(BUILD)/tests/consecutive_launches

cmodel-smoke: cmodel-smoke-bin
	sh ./run_cmodel_smoke.sh

clean:
	rm -rf $(BUILD)

-include $(DEPS)
