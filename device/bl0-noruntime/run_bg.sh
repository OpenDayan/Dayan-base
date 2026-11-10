#!/bin/sh
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

gdb_pid=$1
trap '' SIGINT
[ ! -p /tmp/qemu_stdout ] && mkfifo /tmp/qemu_stdout
umount /tmp/virtual_npu; $QEMU $QEMU_FLAGS </dev/null -s -S >/tmp/qemu_stdout &
qemu_pid=$!
cat /tmp/qemu_stdout | tee -a .qemu_log &
while kill -0 $gdb_pid 2>/dev/null && kill -0 $qemu_pid 2>/dev/null ; do
    sleep 1;
done
kill -9 $qemu_pid 2>/dev/null;true
[ -f .qemu_log ] && rm .qemu_log
