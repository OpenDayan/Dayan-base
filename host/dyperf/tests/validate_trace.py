import json
import math
import sys


with open(sys.argv[1], encoding="utf-8") as trace_file:
    trace = json.load(trace_file)

events = trace["traceEvents"]
assert trace["displayTimeUnit"] == "ns"
assert trace["metadata"]["cycle_ns"] == 1

hart0_pid = 0x10000
hart3_pid = 0x10003

process_sort = {
    event["pid"]: event["args"]["sort_index"]
    for event in events
    if event["name"] == "process_sort_index"
}
assert process_sort[hart0_pid] == 0
assert process_sort[hart3_pid] == 3

thread_names = {
    (event["pid"], event["tid"]): event["args"]["name"]
    for event in events
    if event["name"] == "thread_name"
}
assert thread_names[(hart0_pid, hart0_pid)] == "scheduler"
assert thread_names[(hart0_pid, 1)] == "zeta"
assert thread_names[(hart3_pid, hart3_pid)] == "scheduler"
assert thread_names[(hart3_pid, 7)] == "worker"
assert thread_names[(hart3_pid, 9)] == 'late"child'
assert thread_names[(hart3_pid, 262)] == "worker-2"
assert all(
    (tid == pid) == (name == "scheduler")
    for (pid, tid), name in thread_names.items()
)

thread_sort = {
    (event["pid"], event["tid"]): event["args"]["sort_index"]
    for event in events
    if event["name"] == "thread_sort_index"
}
assert thread_sort[(hart3_pid, 9)] == 0
assert thread_sort[(hart3_pid, hart3_pid)] == 1
assert thread_sort[(hart3_pid, 7)] == 2
assert thread_sort[(hart3_pid, 262)] == 3

worker_wait = next(
    event
    for event in events
    if event.get("pid") == hart3_pid
    and event.get("tid") == 7
    and event["name"] == "WAIT_EVENT"
    and event["ph"] == "B"
)
worker_wait_end = next(
    event
    for event in events
    if event.get("pid") == hart3_pid
    and event.get("tid") == 7
    and event["name"] == "WAIT_EVENT"
    and event["ph"] == "E"
)
assert math.isclose(worker_wait["ts"], 281474976710.640, abs_tol=1e-9)
assert math.isclose(worker_wait_end["ts"], 281474976710.672, abs_tol=1e-9)
assert any(
    event.get("pid") == hart3_pid
    and event.get("tid") == 262
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
