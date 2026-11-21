# backend/simulator/realtime.py
"""
Real-time workload helpers: periodic task sets, utilisation / schedulability
tests, and deadline assignment for aperiodic workloads.

A periodic task (name, period T, worst-case execution time C, phase) releases
a job every T ticks. Each job must finish before the next release (implicit
deadline = release + T).
"""

import math
from typing import Dict, List, Optional, Sequence

from .process import Process

MAX_HORIZON = 2000
MAX_JOBS = 400


def validate_tasks(tasks: Sequence[Dict]) -> Optional[str]:
    """Returns an error message, or None if the task list is valid."""
    if not tasks:
        return "at least one task is required"
    for i, t in enumerate(tasks):
        try:
            period, wcet, phase = int(t["period"]), int(t["wcet"]), int(t.get("phase", 0))
        except (KeyError, TypeError, ValueError):
            return f"task {i}: period and wcet must be integers"
        if period < 1 or wcet < 1:
            return f"task {i}: period and wcet must be >= 1"
        if wcet > period:
            return f"task {i}: wcet ({wcet}) cannot exceed its period ({period})"
        if phase < 0:
            return f"task {i}: phase must be >= 0"
    return None


def expand_periodic(tasks: Sequence[Dict], horizon: int) -> List[Process]:
    """
    One Process per job released in [0, horizon). Jobs carry their task's
    name as `program`, its period, and the implicit deadline release + period.
    PIDs follow release order.
    """
    jobs = []
    for i, t in enumerate(tasks):
        name = str(t.get("name") or f"T{i + 1}")
        period, wcet, phase = int(t["period"]), int(t["wcet"]), int(t.get("phase", 0))
        release = phase
        while release < horizon:
            jobs.append((release, period, i, name, wcet))
            release += period
    jobs.sort(key=lambda j: (j[0], j[1], j[2]))
    return [
        Process(pid=pid, arrival_time=rel, burst_time=wcet, program=name,
                deadline=rel + period, period=period)
        for pid, (rel, period, _, name, wcet) in enumerate(jobs[:MAX_JOBS])
    ]


def hyperperiod(tasks: Sequence[Dict]) -> int:
    h = 1
    for t in tasks:
        h = h * int(t["period"]) // math.gcd(h, int(t["period"]))
    return h


def utilisation(tasks: Sequence[Dict]) -> float:
    return sum(int(t["wcet"]) / int(t["period"]) for t in tasks)


def liu_layland_bound(n: int) -> float:
    """Single-core RMS utilisation bound n(2^(1/n) - 1)."""
    return n * (2 ** (1 / n) - 1) if n > 0 else 0.0


def schedulability(tasks: Sequence[Dict], num_cores: int = 1) -> Dict:
    """
    Utilisation summary for a task set. On one core: RMS is guaranteed when
    U <= the Liu & Layland bound, and EDF iff U <= 1. With several cores the
    per-core load U/m against the same bound is only a heuristic.
    """
    u = utilisation(tasks)
    bound = liu_layland_bound(len(tasks))
    per_core = u / max(num_cores, 1)
    return {
        "utilisation": u,
        "per_core_utilisation": per_core,
        "rms_bound": bound,
        "rms_guaranteed": per_core <= bound,
        "edf_feasible": per_core <= 1.0,
        "exact": num_cores == 1,
        "num_tasks": len(tasks),
    }


def tasks_from_processes(processes: Sequence[Process]) -> List[Dict]:
    """Recover the periodic task set (program, period, max wcet) from expanded jobs."""
    tasks = {}
    for p in processes:
        if p.period is None:
            continue
        key = (p.program, p.period)
        tasks[key] = max(tasks.get(key, 0), p.burst_time)
    return [{"name": k[0], "period": k[1], "wcet": c} for k, c in tasks.items()]


def assign_deadlines(processes: Sequence[Process], slack: float) -> None:
    """In place: deadline = arrival + ceil(burst * slack) for processes without one."""
    for p in processes:
        if p.deadline is None:
            p.deadline = p.arrival_time + max(1, math.ceil(p.burst_time * slack))


PERIODS = [5, 8, 10, 12, 15, 20, 24, 30, 40]


def random_task_set(n_tasks: int, utilisation: float, rng, tolerance: float = 0.03) -> List[Dict]:
    """
    Random implicit-deadline periodic task set (numpy Generator `rng`).
    UUniFast-Discard splits `utilisation` over n tasks (redrawn if any task
    would exceed u_i = 1); wcet = floor(u_i × period) (>= 1). Integer run
    times shift the real utilisation, so sets are redrawn until it lies in
    [utilisation - tolerance, utilisation] — never above the target, so a
    "U = 1.0" single-core set really is feasible for EDF.
    """
    for _ in range(5000):
        utils, remaining = [], utilisation
        for i in range(1, n_tasks):
            nxt = remaining * rng.random() ** (1 / (n_tasks - i))
            utils.append(remaining - nxt)
            remaining = nxt
        utils.append(remaining)
        if max(utils) > 1.0:
            continue
        tasks = []
        for i, u in enumerate(utils):
            period = int(rng.choice(PERIODS))
            tasks.append({"name": f"t{i + 1}", "period": period,
                          "wcet": int(min(period, max(1, math.floor(u * period))))})
        if utilisation - tolerance <= utilisation_of(tasks) <= utilisation + 1e-9:
            return tasks
    raise RuntimeError(f"could not draw a task set with U ≈ {utilisation}")


def utilisation_of(tasks: Sequence[Dict]) -> float:
    return sum(int(t["wcet"]) / int(t["period"]) for t in tasks)
