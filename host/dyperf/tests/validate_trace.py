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

import json
import math
import sys


with open(sys.argv[1], encoding="utf-8") as trace_file:
    trace = json.load(trace_file)

events = trace["traceEvents"]
assert trace["displayTimeUnit"] == "ns"
assert trace["metadata"]["cycle_ns"] == 1

process_ids = {
    event["args"]["name"]: event["pid"]
    for event in events
    if event["name"] == "process_name"
}
hart0_pid = process_ids["cluster 0 hart 0"]
hart3_pid = process_ids["cluster 0 hart 3"]

process_sort = {
    event["pid"]: event["args"]["sort_index"]
    for event in events
    if event["name"] == "process_sort_index"
}
assert process_sort[hart0_pid] == 0
assert process_sort[hart3_pid] == 3
assert process_sort[process_ids["cluster 2 hart 1"]] == (2 << 32) + 1

thread_names = {
    (event["pid"], event["tid"]): event["args"]["name"]
    for event in events
    if event["name"] == "thread_name"
}
assert thread_names[(hart0_pid, hart0_pid)] == "scheduler"
assert thread_names[(hart3_pid, hart3_pid)] == "scheduler"
hart3_tids = {name: tid for (pid, tid), name in thread_names.items() if pid == hart3_pid}
assert set(hart3_tids) == {"scheduler", "worker", 'late"child', "worker-2"}
assert [name for (pid, _), name in thread_names.items() if pid == hart0_pid] == [
    "scheduler", "zeta"
]
name_events = [event for event in events if event["name"] == "thread_name"]
assert len({event["tid"] for event in name_events}) == len(name_events)
assert all(0 < event["tid"] <= 0xFFFFFFFF for event in name_events)
assert len(set(process_ids.values())) == len(process_ids)
assert all(
    (tid == pid) == (name == "scheduler")
    for (pid, tid), name in thread_names.items()
)

thread_sort = {
    (event["pid"], event["tid"]): event["args"]["sort_index"]
    for event in events
    if event["name"] == "thread_sort_index"
}
assert thread_sort[(hart3_pid, hart3_pid)] == 0
assert thread_sort[(hart3_pid, hart3_tids["worker"])] == 1
assert thread_sort[(hart3_pid, hart3_tids['late"child'])] == 2
assert thread_sort[(hart3_pid, hart3_tids["worker-2"])] == 3

first_cycles = {
    (event["pid"], event["tid"]): event["args"]["counter"]
    for event in events
    if event["ph"] == "B" and event["name"] == "READY"
}
for cluster in (0, 2):
    pid = process_ids[f"cluster {cluster} hart 1"]
    tracks = sorted(
        (key for key in thread_names if key[0] == pid), key=thread_sort.__getitem__
    )
    assert [thread_sort[key] for key in tracks] == list(range(len(tracks)))
    assert [thread_names[key] for key in tracks[:3]] == [
        "scheduler", "profile-flush", "rpc-receive"
    ]
    assert len(tracks) == 3 + 300 * 7
    for round_index in range(300):
        family_tracks = tracks[3 + round_index * 7:3 + (round_index + 1) * 7]
        assert [thread_names[key] for key in family_tracks] == [
            "z-root", "load-a", "load-a", "load-a", "load-b", "a-root", "post"
        ]
        assert [first_cycles[key] for key in family_tracks] == [
            100 + round_index * 100 + offset for offset in (0, 4, 5, 6, 3, 1, 2)
        ]

worker_wait = next(
    event
    for event in events
    if event.get("pid") == hart3_pid
    and event.get("tid") == hart3_tids["worker"]
    and event["name"] == "WAIT_EVENT"
    and event["ph"] == "B"
)
worker_wait_end = next(
    event
    for event in events
    if event.get("pid") == hart3_pid
    and event.get("tid") == hart3_tids["worker"]
    and event["name"] == "WAIT_EVENT"
    and event["ph"] == "E"
)
assert math.isclose(worker_wait["ts"], 281474976710.640, abs_tol=1e-9)
assert math.isclose(worker_wait_end["ts"], 281474976710.672, abs_tol=1e-9)
assert any(
    event.get("pid") == hart3_pid
    and event.get("tid") == hart3_tids["worker-2"]
    and event["name"] == "RUNNING"
    and event["ph"] == "B"
    for event in events
)
scheduler_reuse_begin = next(
    event
    for event in events
    if event.get("pid") == hart3_pid
    and event.get("tid") == hart3_pid
    and event["name"] == "RUNNING"
    and event["ph"] == "B"
    and event["args"]["counter"] == 0x30
)
scheduler_reuse_end = next(
    event
    for event in events
    if event.get("pid") == hart3_pid
    and event.get("tid") == hart3_pid
    and event["name"] == "RUNNING"
    and event["ph"] == "E"
    and event["args"]["counter"] == 0x38
)
assert scheduler_reuse_begin["ts"] < scheduler_reuse_end["ts"]
assert any(
    event.get("pid") == hart3_pid
    and event.get("tid") == hart3_pid
    and event["name"] == "RUNNING"
    and event["ph"] == "B"
    for event in events
)
