# Included by device/Makefile so tests and libnpu-os.a share the exact ABI/config.
DEVICE_TEST_BUILD ?= $(ROOT)/build/device_tests
DEVICE_TEST_NAMES := $(notdir $(patsubst %/,%,$(wildcard $(ROOT)/device_tests/*_test/)))
DEVICE_TEST_OBJECTS :=

.PHONY: device-tests device-tests-clean
device-tests:

define device_test_rules
DEVICE_TEST_SOURCES_$(1) := $(sort $(wildcard $(ROOT)/device_tests/$(1)/*.c \
    $(ROOT)/device_tests/$(1)/*.cpp $(ROOT)/device_tests/$(1)/*.cc \
    $(ROOT)/device_tests/$(1)/*.S))
DEVICE_TEST_OBJECTS_$(1) := $$(patsubst $(ROOT)/device_tests/%,$(DEVICE_TEST_BUILD)/%.o,$$(DEVICE_TEST_SOURCES_$(1)))
DEVICE_TEST_OBJECTS += $$(DEVICE_TEST_OBJECTS_$(1))

.PHONY: device-test-$(1)
device-tests: device-test-$(1)
device-test-$(1): $(DEVICE_TEST_BUILD)/$(1)/$(1).elf $(DEVICE_TEST_BUILD)/$(1)/$(1).bin

$(DEVICE_TEST_BUILD)/$(1)/$(1).elf: $$(DEVICE_TEST_OBJECTS_$(1)) $(LIB) \
    linker.ld $(ROOT)/device_tests/rules.mk $(ROOT)/device_tests/$(1)
	@mkdir -p $$(dir $$@)
	$(CLANG) $(LDFLAGS) -Wl,-Map,$(DEVICE_TEST_BUILD)/$(1)/$(1).map,--gc-sections \
		-Wl,--defsym=CONFIG_REQUIRE_TESTS=1 \
		$$(DEVICE_TEST_OBJECTS_$(1)) $(LIB) -o $$@

$(DEVICE_TEST_BUILD)/$(1)/$(1).bin: $(DEVICE_TEST_BUILD)/$(1)/$(1).elf
	$(OBJCOPY) -O binary $$< $$@
endef

$(foreach test,$(DEVICE_TEST_NAMES),$(eval $(call device_test_rules,$(test))))

$(DEVICE_TEST_BUILD)/%.c.o: $(ROOT)/device_tests/%.c $(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	$(CLANG) $(CPPFLAGS) -I$(dir $<) $(CFLAGS) $(RVT_ISA_FLAGS) -MMD -MP -c $< -o $@

$(DEVICE_TEST_BUILD)/%.cpp.o: $(ROOT)/device_tests/%.cpp $(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	$(CLANGXX) $(CPPFLAGS) -I$(dir $<) $(filter-out -std=c11,$(CFLAGS)) \
		$(RVT_ISA_FLAGS) -std=c++17 -fno-exceptions -fno-rtti -MMD -MP -c $< -o $@

$(DEVICE_TEST_BUILD)/%.cc.o: $(ROOT)/device_tests/%.cc $(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	$(CLANGXX) $(CPPFLAGS) -I$(dir $<) $(filter-out -std=c11,$(CFLAGS)) \
		$(RVT_ISA_FLAGS) -std=c++17 -fno-exceptions -fno-rtti -MMD -MP -c $< -o $@

$(DEVICE_TEST_BUILD)/%.S.o: $(ROOT)/device_tests/%.S $(CONFIG_STAMP)
	@mkdir -p $(dir $@)
	$(CLANG) $(CPPFLAGS) -I$(dir $<) $(CFLAGS) -MMD -MP -c $< -o $@

device-tests-clean:
	rm -rf $(DEVICE_TEST_BUILD)

-include $(DEVICE_TEST_OBJECTS:.o=.d)
