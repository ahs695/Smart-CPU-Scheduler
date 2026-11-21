import random
from typing import List

from .process import Process


class WorkloadGenerator:
    """
    Generates synthetic workloads for CPU scheduling experiments.
    Supports CPU-bound, IO-bound, mixed, and random workloads.
    """

    # ------------------------------------------------------------

    @staticmethod
    def cpu_bound(
        num_processes: int,
        arrival_gap: int = 3,
        burst_range=(20, 50)
    ) -> List[Process]:

        processes = []

        for pid in range(1, num_processes + 1):

            arrival = pid * arrival_gap
            burst = random.randint(*burst_range)

            processes.append(
                Process(pid=pid, arrival_time=arrival, burst_time=burst)
            )

        return processes

    # ------------------------------------------------------------

    @staticmethod
    def io_bound(
        num_processes: int,
        arrival_gap: int = 1,
        burst_range=(1, 8)
    ) -> List[Process]:

        processes = []

        for pid in range(1, num_processes + 1):

            arrival = pid * arrival_gap
            burst = random.randint(*burst_range)

            processes.append(
                Process(pid=pid, arrival_time=arrival, burst_time=burst)
            )

        return processes

    # ------------------------------------------------------------

    @staticmethod
    def mixed(
        num_processes: int,
        cpu_ratio: float = 0.3
    ) -> List[Process]:

        processes = []

        cpu_count = int(num_processes * cpu_ratio)

        for pid in range(1, num_processes + 1):

            arrival = random.randint(0, num_processes)

            if pid <= cpu_count:
                burst = random.randint(20, 50)
            else:
                burst = random.randint(1, 8)

            processes.append(
                Process(pid=pid, arrival_time=arrival, burst_time=burst)
            )

        return sorted(processes, key=lambda p: p.arrival_time)

    # ------------------------------------------------------------

    @staticmethod
    def poisson_arrivals(
        num_processes: int,
        rate: float = 0.5,
        burst_range=(5, 30)
    ) -> List[Process]:

        processes = []

        current_time = 0

        for pid in range(1, num_processes + 1):

            inter_arrival = random.expovariate(rate)
            current_time += int(inter_arrival)

            burst = random.randint(*burst_range)

            processes.append(
                Process(pid=pid, arrival_time=current_time, burst_time=burst)
            )

        return processes

    # ------------------------------------------------------------

    @staticmethod
    def random_workload(
        num_processes: int,
        max_arrival: int = 50,
        burst_range=(1, 50)
    ) -> List[Process]:

        processes = []

        for pid in range(1, num_processes + 1):

            arrival = random.randint(0, max_arrival)
            burst = random.randint(*burst_range)

            processes.append(
                Process(pid=pid, arrival_time=arrival, burst_time=burst)
            )

        return sorted(processes, key=lambda p: p.arrival_time)


# ------------------------------------------------------------
# Program-based workloads (numpy Generator driven)
#
# Each workload runs a handful of "programs" (think executables / users).
# Every program has its own typical burst; its jobs vary around it
# (x0.7-1.3) with occasional outliers. A real OS knows which program a job
# belongs to, so the Hybrid scheduler can predict a job's burst from that
# program's finished jobs. Burst *values* never leak to the agents.
# ------------------------------------------------------------

BURST_PATTERNS = ["cpu_heavy", "io_heavy", "mixed", "random", "phased"]
BURST_RANGE = {"cpu_heavy": (20, 50), "io_heavy": (1, 8), "random": (1, 50)}
PHASE_RANGES = [(1, 6), (8, 15), (20, 45)]   # interactive, medium, batch
JOB_NOISE = (0.7, 1.3)
OUTLIER_PROB = 0.1


def _program_means(pattern: str, k: int, rng) -> List[float]:
    if pattern in BURST_RANGE:
        lo, hi = BURST_RANGE[pattern]
        return [float(rng.uniform(lo, hi)) for _ in range(k)]
    if pattern == "mixed":
        # ~30% CPU-bound programs, the rest I/O-bound (at least one of each)
        n_cpu = max(1, int(round(0.3 * k)))
        return [float(rng.uniform(20, 50)) for _ in range(n_cpu)] + \
               [float(rng.uniform(1, 8)) for _ in range(k - n_cpu)]
    raise ValueError(pattern)


def _job_burst(mean: float, lo: int, hi: int, rng) -> int:
    if rng.random() < OUTLIER_PROB:
        return int(rng.integers(lo, hi + 1))
    return int(min(max(round(mean * rng.uniform(*JOB_NOISE)), lo), hi))


def sample_jobs(pattern: str, n: int, rng):
    """Returns (bursts, programs) for n jobs in arrival order."""
    if pattern == "phased":
        # Each phase has its own programs; phases switch as a Markov chain
        stay = rng.uniform(0.75, 0.95)
        phase_programs, pid = [], 1
        for lo, hi in PHASE_RANGES:
            progs = []
            for _ in range(int(rng.integers(2, 4))):
                progs.append((pid, float(rng.uniform(lo, hi))))
                pid += 1
            phase_programs.append(progs)
        phase = int(rng.integers(len(PHASE_RANGES)))
        bursts, programs = [], []
        for i in range(n):
            if i > 0 and rng.random() > stay:
                phase = int(rng.choice([r for r in range(len(PHASE_RANGES)) if r != phase]))
            prog, mean = phase_programs[phase][int(rng.integers(len(phase_programs[phase])))]
            lo, hi = PHASE_RANGES[phase]
            bursts.append(_job_burst(mean, lo, hi, rng))
            programs.append(prog)
        return bursts, programs

    k = int(rng.integers(3, 9))
    means = _program_means(pattern, k, rng)
    lo, hi = BURST_RANGE.get(pattern, (1, 50))
    programs = [int(p) for p in rng.integers(0, k, n)]
    bursts = []
    for p in programs:
        if pattern == "mixed":
            lo, hi = (20, 50) if means[p] >= 20 else (1, 8)
        bursts.append(_job_burst(means[p], lo, hi, rng))
    return bursts, [p + 1 for p in programs]


def native_arrivals(pattern: str, n: int, rng) -> List[int]:
    """Arrival pattern each workload type has used historically."""
    if pattern == "cpu_heavy":
        return [3 * (i + 1) for i in range(n)]
    if pattern == "io_heavy":
        return [i + 1 for i in range(n)]
    if pattern == "mixed":
        return sorted(int(a) for a in rng.integers(0, n + 1, n))
    if pattern == "random":
        return sorted(int(a) for a in rng.integers(0, 51, n))
    if pattern == "phased":
        gaps = rng.exponential(1 / 0.2, n)
        gaps[0] = 0.0
        return [int(t) for t in gaps.cumsum()]
    raise ValueError(pattern)


def build_processes(arrivals, bursts, programs) -> List[Process]:
    return [
        Process(pid=i + 1, arrival_time=a, burst_time=b, program=g)
        for i, (a, b, g) in enumerate(zip(arrivals, bursts, programs))
    ]


def sample_training_workload(rng, num_cores: int) -> List[Process]:
    """
    Fresh, randomised workload for RL training (never the seeded evaluation sets).

    - burst pattern: any of BURST_PATTERNS, with per-workload programs
    - size: 10-60 processes
    - arrivals: mostly Poisson at a load factor rho in [0.6, 1.3] relative to the
      core count (so multi-core runs still see contention); otherwise the type's
      native pattern, a single batch at t=0, or a light trickle.
    - 10% of workloads carry no program labels, so policies also handle that case.
    """
    pattern = BURST_PATTERNS[int(rng.integers(len(BURST_PATTERNS)))]
    n = int(rng.integers(10, 61))
    bursts, programs = sample_jobs(pattern, n, rng)
    mean_burst = max(1.0, sum(bursts) / n)

    mode = rng.random()
    if mode < 0.65:
        rate = rng.uniform(0.6, 1.3) * num_cores / mean_burst
        gaps = rng.exponential(1.0 / rate, n)
        gaps[0] = 0.0
        arrivals = [int(t) for t in gaps.cumsum()]
    elif mode < 0.8:
        arrivals = native_arrivals(pattern, n, rng)
    elif mode < 0.9:
        arrivals = [0] * n
    else:
        rate = rng.uniform(0.3, 0.6) * num_cores / mean_burst
        gaps = rng.exponential(1.0 / rate, n)
        gaps[0] = 0.0
        arrivals = [int(t) for t in gaps.cumsum()]

    if rng.random() < 0.1:
        programs = [None] * n
    return build_processes(arrivals, bursts, programs)
