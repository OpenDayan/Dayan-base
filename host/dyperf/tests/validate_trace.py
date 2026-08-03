import json
import math
import sys


with open(sys.argv[1], encoding="utf-8") as trace_file:
    trace = json.load(trace_file)

events = trace["traceEvents"]
assert trace["displayTimeUnit"] == "ns"
assert trace["metadata"]["cycle_ns"] == 1

process_sort = {
    event["pid"]: event["args"]["sort_index"]
    for event in events
    if event["name"] == "process_sort_index"
}
assert process_sort[1] == 0
assert process_sort[4] == 3

thread_names = {
    (event["pid"], event["tid"]): event["args"]["name"]
    for event in events
    if event["name"] == "thread_name"
}
assert thread_names[(4, 0)] == "scheduler"
assert thread_names[(4, 7)] == "worker"
assert thread_names[(4, 9)] == 'late"child'

thread_sort = {
    (event["pid"], event["tid"]): event["args"]["sort_index"]
    for event in events
    if event["name"] == "thread_sort_index"
}
assert thread_sort[(4, 9)] == 0
assert thread_sort[(4, 0)] == 1
assert thread_sort[(4, 7)] == 2

worker_wait = next(
    event
    for event in events
    if event.get("pid") == 4
    and event.get("tid") == 7
    and event["name"] == "WAIT_EVENT"
    and event["ph"] == "B"
)
worker_wait_end = next(
    event
    for event in events
    if event.get("pid") == 4
    and event.get("tid") == 7
    and event["name"] == "WAIT_EVENT"
    and event["ph"] == "E"
)
assert math.isclose(worker_wait["ts"], 4294967.280, abs_tol=1e-9)
assert math.isclose(worker_wait_end["ts"], 4294967.312, abs_tol=1e-9)
assert any(
    event.get("pid") == 4
    and event.get("tid") == 0
    and event["name"] == "RUNNING"
    and event["ph"] == "B"
    for event in events
)
