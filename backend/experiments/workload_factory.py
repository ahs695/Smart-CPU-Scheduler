# backend/experiments/workload_factory.py
"""
Seeded workload factory.

Single source of truth for generating reproducible process lists.
All experiment scripts must use this module to guarantee that every
scheduler is evaluated on identical inputs for a given (workload_type, run_id).
"""

import random
import numpy as np
from typing import List

from backend.simulator.workload_generator import WorkloadGenerator
from backend.simulator.process import Process


# Supported workload type labels
WORKLOAD_TYPES = ["cpu_heavy", "io_heavy", "mixed", "random"]


def generate_workload(
    workload_type: str,
    n_processes: int = 30,
    seed: int = 0
) -> List[Process]:
    """
    Generate a reproducible list of Process objects.

    The (workload_type, seed) pair always produces the same workload,
    ensuring fair head-to-head comparison across all schedulers.

    Args:
        workload_type: One of 'cpu_heavy', 'io_heavy', 'mixed', 'random'
        n_processes:   Number of processes to generate
        seed:          RNG seed (use run_id as seed for reproducibility)

    Returns:
        A fresh list of Process objects (all runtime fields at default values).
    """
    random.seed(seed)
    np.random.seed(seed)

    if workload_type == "cpu_heavy":
        return WorkloadGenerator.cpu_bound(n_processes)

    elif workload_type == "io_heavy":
        return WorkloadGenerator.io_bound(n_processes)

    elif workload_type == "mixed":
        return WorkloadGenerator.mixed(n_processes)

    elif workload_type == "random":
        return WorkloadGenerator.random_workload(n_processes)

    else:
        raise ValueError(
            f"Unknown workload_type '{workload_type}'. "
            f"Choose from: {WORKLOAD_TYPES}"
        )
