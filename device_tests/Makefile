TESTS := $(sort $(patsubst ./%/meta.json,%,$(shell find . -type f -name meta.json)))

.PHONY: all clean $(TESTS)

all:
	$(MAKE) -C ../device device-tests

$(TESTS):
	$(MAKE) -C ../device device-test-$@

clean:
	$(MAKE) -C ../device device-tests-clean
