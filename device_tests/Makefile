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

TESTS := $(sort $(patsubst ./%/meta.json,%,$(shell find . -type f -name meta.json)))

.PHONY: all clean $(TESTS)

all:
	$(MAKE) -C ../device device-tests

$(TESTS):
	$(MAKE) -C ../device device-test-$@

clean:
	$(MAKE) -C ../device device-tests-clean
