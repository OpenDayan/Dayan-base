#!/usr/bin/env python3
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

"""Measure scheduler_yield_bench notification intervals from dyperf records."""

import argparse
import json
import math
import statistics
from pathlib import Path


def wait_intervals(events, pid):
    tids = {
        event["tid"] for event in events
        if event.get("ph") == "M" and event.get("name") == "thread_name"
        and event.get("pid") == pid
        and event["args"]["name"] == "notify-bench"
    }
    if len(tids) != 1:
        raise ValueError(f"expected one notify-bench thread on pid {pid}")
    tid = tids.pop()
    intervals = []
    waiting = None
    for event in events:
        if (event.get("pid"), event.get("tid"), event.get("ph")) != (pid, tid, "B"):
            continue
        counter = event["args"]["counter"]
        if event["name"] == "WAIT_PRODUCE":
            if waiting is not None:
                raise ValueError("repeated wait record; check profile buffer flushes")
            waiting = counter
        elif event["name"] == "RUNNING" and waiting is not None:
            intervals.append((waiting, counter))
            waiting = None
    if waiting is not None:
        raise ValueError("trace ends with an incomplete resource wait")
    return intervals


def delta(end, start):
    # Profile counters are 48-bit. Cross-Hart subtraction assumes aligned clocks.
    return ((end - start + (1 << 47)) % (1 << 48)) - (1 << 47)


def summarize(values):
    ordered = sorted(values)
    if not ordered or ordered[0] < 0:
        raise ValueError("missing samples or invalid counter order/alignment")
    return {
        "count": len(ordered), "min": ordered[0],
        "median": statistics.median(ordered), "mean": statistics.mean(ordered),
        "p95": ordered[math.ceil(len(ordered) * 0.95) - 1],
        "p99": ordered[math.ceil(len(ordered) * 0.99) - 1], "max": ordered[-1],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--iterations", type=int, default=128)
    parser.add_argument("--warmup", type=int, default=64)
    parser.add_argument("--cluster", type=int, default=0)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.iterations < 1 or args.warmup < 0:
        parser.error("iterations must be positive and warmup nonnegative")
    events = json.loads(args.trace.read_text())["traceEvents"]
    pids = {
        event["args"]["name"]: event["pid"] for event in events
        if event.get("ph") == "M" and event.get("name") == "process_name"
    }
    waits = [wait_intervals(events, pids[f"cluster {args.cluster} hart {hart}"])
             for hart in range(4)]
    total = args.warmup + args.iterations
    if [len(items) for items in waits] != [total + 2, total + 2, 2, 2]:
        raise ValueError("unexpected wait counts; check iterations, warmup and trace completeness")
    # TX: notify*total, done, LSU drain. RX: reserve done, notify*total, LSU drain.
    send = waits[0][:total]
    receive = waits[1][1:total + 1]
    samples = range(args.warmup, total)
    report = {
        "trace": str(args.trace), "unit": "cycles", "profile_enabled": True,
        "iterations": args.iterations, "warmup_discarded": args.warmup,
        "boundary": "Hart0 RUNNING after wait-free -> Hart1 RUNNING after wait-ready",
        "caveat": "Includes sender resume/Produce API and profiling; not an instruction-level Produce timestamp. Cross-Hart counters must be aligned.",
        "notify_dispatch_cycles": summarize([
            delta(receive[i][1], send[i][1]) for i in samples]),
        "receiver_wait_cycles": summarize([
            delta(receive[i][1], receive[i][0]) for i in samples]),
        "sender_round_trip_cycles": summarize([
            delta(send[i + 1][1], send[i][1]) for i in range(args.warmup, total - 1)
        ]) if args.iterations > 1 else None,
        "receiver_wait_record_precedes_sender_dispatch": sum(
            delta(send[i][1], receive[i][0]) >= 0 for i in samples),
        "idle_hart_wait_counts": [len(waits[2]), len(waits[3])],
    }
    text = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.write_text(text)
    print(text, end="")


if __name__ == "__main__":
    main()
