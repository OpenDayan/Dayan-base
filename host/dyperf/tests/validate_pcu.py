import json
import math
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
intervals = [event for event in events if event["ph"] == "X"]
assert len(intervals) == 2
first, second = intervals
assert first["ts"] == 2 and first["dur"] == 2
assert second["ts"] == 4 and second["dur"] == 4
assert first["args"]["cycles"] == 1000
assert second["args"]["cycles"] == 2000
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
pid = 0x10000 + (2 << 32) + 1
for name, value in expected.items():
    assert math.isclose(first["args"][name], value)
    assert second["args"][name] == 0
    counters = [e for e in events if e["name"] == name and e["ph"] == "C"]
    assert [e["ts"] for e in counters] == [2, 4, 8]
    assert [e["args"]["value"] for e in counters] == [value, 0, 0]
    assert all(e["pid"] == pid for e in counters)
assert all(e["tid"] == 1 << 40 for e in intervals + snapshots)
