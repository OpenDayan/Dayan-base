import json
import sys


with open(sys.argv[1], encoding="utf-8") as trace_file:
    trace = json.load(trace_file)

assert trace["metadata"]["pcu_enabled"] is True
assert trace["metadata"]["clock_hz"] == 500_000_000
assert trace["metadata"]["cycle_ns"] == 2
events = trace["traceEvents"]
snapshots = [event for event in events if event["name"] == "PCU snapshot"]
assert [event["args"]["cycle"] for event in snapshots] == [1000, 2000, 4000]
assert [event["ts"] for event in snapshots] == [2, 4, 8]
assert all(len(event["args"]["counters"]) == 112 for event in snapshots)
assert not any(event["ph"] == "X" for event in events)
assert not any(event["name"] == "PCU interval" for event in events)
expected = {
    "URAM read mean (%)": 50,
    "URAM read variance (%^2)": 2500,
    "URAM read stddev (%)": 50,
    "URAM write mean (%)": 25,
    "URAM write variance (%^2)": 0,
    "URAM write stddev (%)": 0,
    "MAT utilization (%)": 75,
    "LSU read (GB/s)": 4,
    "LSU write (GB/s)": 8,
    "VEC group 0 IPC": 0.5,
    "VEC group 1 IPC": 0.25,
    "VEC total IPC": 0.75,
    "Scalar IPC": 1.5,
}
pid = next(event["pid"] for event in events if event["name"] == "process_name")
pcu_tid = next(
    event["tid"] for event in events
    if event["name"] == "thread_name" and event["args"]["name"] == "PCU snapshots"
)
assert 0 < pcu_tid <= 0xFFFFFFFF and pcu_tid != pid
for name, value in expected.items():
    counters = [e for e in events if e["name"] == name and e["ph"] == "C"]
    assert [e["ts"] for e in counters] == [2, 4, 8]
    assert [e["args"]["value"] for e in counters] == [value, 0, 0]
    assert all(e["pid"] == pid for e in counters)
    assert all(e["tid"] == pcu_tid for e in counters)
assert all(e["tid"] == pcu_tid for e in snapshots)
