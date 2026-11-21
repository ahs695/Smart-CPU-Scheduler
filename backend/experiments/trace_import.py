# backend/experiments/trace_import.py
"""
Import real cluster traces as simulator workloads.

Supported formats (CSV, no header, as published):

  google   Google cluster-data 2011 `task_events` table, 13 columns:
             0 timestamp (µs) · 1 missing info · 2 job ID · 3 task index ·
             4 machine ID · 5 event type · 6 user · 7 scheduling class ·
             8 priority · 9 CPU request · 10 memory request · 11 disk request ·
             12 different-machines restriction
           Event types: 0 SUBMIT, 1 SCHEDULE, 4 FINISH (others ignored).
           A task becomes one process: arrival = SUBMIT, burst = FINISH − SCHEDULE,
           program = user. Scheduling class 1-3 marks latency-sensitive work and
           gets a deadline (tighter for higher classes); class 0 (batch) gets none.

  alibaba  Alibaba cluster-trace-v2018 `batch_task.csv`, 9 columns:
             0 task_name · 1 instance_num · 2 job_name · 3 task_type · 4 status ·
             5 start_time (s) · 6 end_time (s) · 7 plan_cpu · 8 plan_mem
           Only rows with status "Terminated". arrival = start_time,
           burst = end_time − start_time, program = job_name.

Real traces span hours and millions of tasks, so a window of `max_jobs`
consecutive tasks (by arrival, after `offset`) is taken and time is rescaled
with ONE factor for durations and arrival gaps (the 95th-percentile duration
maps to 50 ticks), which preserves the trace's load. Bursts are clipped to
[1, 50] ticks — the range the RL schedulers and LSTM were trained on.
Optionally `load` compresses arrivals so the offered load ≈ load × num_cores.
"""

import csv
import io
import math
from typing import Dict, List, Optional, Tuple

from backend.simulator.process import Process

MAX_BURST_TICKS = 50
MAX_ROWS = 2_000_000                     # safety cap on rows read from one file
GOOGLE_COLUMNS, ALIBABA_COLUMNS = 13, 9
GOOGLE_CLASS_SLACK = {3: 2.0, 2: 3.0, 1: 5.0}   # deadline = arrival + slack × burst


class TraceFormatError(ValueError):
    pass


# ------------------------------------------------------------
# Parsing
# ------------------------------------------------------------

def _rows(text: str):
    reader = csv.reader(io.StringIO(text))
    for i, row in enumerate(reader):
        if i >= MAX_ROWS:
            break
        if row and any(cell.strip() for cell in row):
            yield [cell.strip() for cell in row]


def detect_format(text: str) -> str:
    for row in _rows(text):
        if len(row) == GOOGLE_COLUMNS:
            return "google"
        if len(row) == ALIBABA_COLUMNS:
            return "alibaba"
        raise TraceFormatError(
            f"Unrecognised trace: first row has {len(row)} columns "
            f"(expected {GOOGLE_COLUMNS} for Google task_events or {ALIBABA_COLUMNS} for Alibaba batch_task)"
        )
    raise TraceFormatError("The file is empty")


def parse_google(text: str) -> List[Dict]:
    """Completed tasks as {arrival, duration, program, sched_class} (times in µs)."""
    submit, schedule, tasks = {}, {}, []
    for n, row in enumerate(_rows(text), start=1):
        if len(row) != GOOGLE_COLUMNS:
            raise TraceFormatError(f"Line {n}: expected {GOOGLE_COLUMNS} columns, got {len(row)}")
        try:
            ts, event = int(row[0]), int(row[5])
            key = (row[2], row[3])
            sched_class = int(row[7]) if row[7] else 0
        except ValueError:
            raise TraceFormatError(f"Line {n}: timestamp, event type and scheduling class must be integers")
        if event == 0:
            submit.setdefault(key, ts)
        elif event == 1:
            schedule[key] = ts
        elif event == 4 and key in schedule:
            duration = ts - schedule.pop(key)
            if duration > 0:
                tasks.append({"arrival": submit.pop(key, ts - duration), "duration": duration,
                              "program": row[6] or None, "sched_class": sched_class})
    return tasks


def parse_alibaba(text: str) -> List[Dict]:
    """Terminated tasks as {arrival, duration, program} (times in s)."""
    tasks = []
    for n, row in enumerate(_rows(text), start=1):
        if len(row) != ALIBABA_COLUMNS:
            raise TraceFormatError(f"Line {n}: expected {ALIBABA_COLUMNS} columns, got {len(row)}")
        if row[4] != "Terminated":
            continue
        try:
            start, end = int(float(row[5])), int(float(row[6]))
        except ValueError:
            raise TraceFormatError(f"Line {n}: start_time and end_time must be numbers")
        if end > start:
            tasks.append({"arrival": start, "duration": end - start, "program": row[2] or None})
    return tasks


# ------------------------------------------------------------
# Conversion to processes
# ------------------------------------------------------------

def _percentile(values: List[float], q: float) -> float:
    s = sorted(values)
    return s[min(len(s) - 1, int(math.ceil(q * len(s))) - 1)]


def to_processes(
    tasks: List[Dict],
    max_jobs: int = 60,
    offset: int = 0,
    load: Optional[float] = None,
    num_cores: int = 1,
    slack: Optional[float] = None,
) -> Tuple[List[Process], Dict]:
    if not tasks:
        raise TraceFormatError("No completed tasks found in the trace")
    window = sorted(tasks, key=lambda t: t["arrival"])[offset: offset + max_jobs]
    if not window:
        raise TraceFormatError(f"offset {offset} is past the {len(tasks)} completed tasks in the trace")

    scale = max(_percentile([t["duration"] for t in window], 0.95) / MAX_BURST_TICKS, 1e-9)
    t0 = window[0]["arrival"]
    bursts = [min(max(int(round(t["duration"] / scale)), 1), MAX_BURST_TICKS) for t in window]
    arrivals = [(t["arrival"] - t0) / scale for t in window]

    if load and len(window) > 1 and arrivals[-1] > 0:
        # Compress/stretch arrival gaps so total work / span ≈ load × cores
        target_span = sum(bursts) / (load * num_cores)
        factor = target_span / arrivals[-1]
        arrivals = [a * factor for a in arrivals]

    processes = []
    for pid, (t, a, b) in enumerate(zip(window, arrivals, bursts)):
        p = Process(pid=pid, arrival_time=int(round(a)), burst_time=b,
                    program=None if t["program"] is None else str(t["program"]))
        cls_slack = GOOGLE_CLASS_SLACK.get(t.get("sched_class"))
        job_slack = cls_slack if cls_slack is not None else slack
        if job_slack is not None:
            p.deadline = p.arrival_time + max(1, math.ceil(b * job_slack))
        processes.append(p)

    stats = {
        "completed_tasks_in_file": len(tasks),
        "imported": len(processes),
        "seconds_per_tick": scale / (1e6 if window and "sched_class" in window[0] else 1),
        "clipped_bursts": sum(1 for t in window if t["duration"] / scale > MAX_BURST_TICKS),
        "with_deadline": sum(p.deadline is not None for p in processes),
        "programs": len({p.program for p in processes}),
    }
    return processes, stats


def import_trace(text: str, fmt: str = "auto", **kwargs) -> Tuple[List[Process], str, Dict]:
    fmt = detect_format(text) if fmt == "auto" else fmt
    if fmt == "google":
        tasks = parse_google(text)
    elif fmt == "alibaba":
        tasks = parse_alibaba(text)
    else:
        raise TraceFormatError(f"Unknown format '{fmt}' (use auto, google or alibaba)")
    processes, stats = to_processes(tasks, **kwargs)
    return processes, fmt, stats
