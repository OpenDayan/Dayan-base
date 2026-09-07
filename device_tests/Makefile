TESTS := $(notdir $(patsubst %/,%,$(wildcard *_test/)))

.PHONY: all clean $(TESTS)

all:
	$(MAKE) -C ../device device-tests

$(TESTS):
	$(MAKE) -C ../device device-test-$@

clean:
	$(MAKE) -C ../device device-tests-clean
