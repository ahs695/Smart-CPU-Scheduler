# backend/experiments/workload_factory.py
"""
Seeded workload factory.

Single source of truth for generating reproducible process lists.
All experiment scripts must use this module to guarantee that every
scheduler is evaluated on identical inputs for a given (workload_type, run_id).
"""

from typing import List

import numpy as np

from backend.simulator.process import Process
from backend.simulator.workload_generator import (
    BURST_PATTERNS, build_processes, native_arrivals, sample_jobs,
)

# Supported workload type labels
WORKLOAD_TYPES = list(BURST_PATTERNS)


def generate_workload(
    workload_type: str,
    n_processes: int = 30,
    seed: int = 0
) -> List[Process]:
    """
    Generate a reproducible list of Process objects.

    The (workload_type, seed) pair always produces the same workload, so every
    scheduler is compared on identical inputs. Each workload runs a few
    programs with their own typical bursts (see workload_generator.sample_jobs).
    """
    if workload_type not in WORKLOAD_TYPES:
        raise ValueError(f"Unknown workload_type '{workload_type}'. Choose from: {WORKLOAD_TYPES}")
    rng = np.random.default_rng(seed)
    bursts, programs = sample_jobs(workload_type, n_processes, rng)
    arrivals = native_arrivals(workload_type, n_processes, rng)
    return build_processes(arrivals, bursts, programs)
