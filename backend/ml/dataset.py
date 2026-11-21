# backend/ml/dataset.py
"""
Datasets for the LSTM burst predictor.

Samples come from the same workload generator the RL agents train on. Every
job's burst is predicted from the bursts of earlier jobs of the *same program*
(normalised, left-padded; see history_window in lstm_model.py). At run time
the Hybrid env only knows the bursts of jobs that have already finished, so
the most recent 0-2 same-program jobs are randomly dropped from each history
to mimic jobs still running. Workloads without program labels use the
history of all earlier jobs.
"""

from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from backend.ml.lstm_model import history_window, normalize_burst
from backend.simulator.workload_generator import BURST_PATTERNS, sample_jobs


class BurstDataset(Dataset):
    """
    PyTorch dataset for CPU burst prediction.

    Each sample:
        input  -> SEQ_LENGTH normalised previous bursts
        target -> normalised next burst
    """

    def __init__(self, sequences: List[List[float]], targets: List[float]):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class DatasetBuilder:
    """Builds burst-prediction samples from generated workloads."""

    @staticmethod
    def histories(bursts: Sequence[int], programs: Sequence, rng) -> List[List[int]]:
        """Known same-program history (raw bursts) for each job, in arrival order."""
        seen: Dict[object, List[int]] = {}
        out = []
        for burst, prog in zip(bursts, programs):
            past = seen.setdefault(prog, [])
            drop = int(rng.integers(0, 3))
            out.append(past[: max(0, len(past) - drop)])
            past.append(burst)
        return out

    @staticmethod
    def generate(
        num_workloads: int,
        seed: int,
        patterns: Optional[Sequence[str]] = None,
        min_len: int = 20,
        max_len: int = 80,
    ) -> Dict[str, Tuple[List[List[float]], List[float], List[List[int]]]]:
        """
        Generate samples grouped by burst pattern.

        Returns {pattern: (X, y, raw_histories)} where raw_histories holds the
        un-normalised known history of each job (for naive baselines).
        """
        rng = np.random.default_rng(seed)
        patterns = list(patterns or BURST_PATTERNS)
        out = {p: ([], [], []) for p in patterns}
        for _ in range(num_workloads):
            pattern = patterns[int(rng.integers(len(patterns)))]
            bursts, programs = sample_jobs(pattern, int(rng.integers(min_len, max_len + 1)), rng)
            if rng.random() < 0.1:
                programs = [None] * len(bursts)
            for burst, hist in zip(bursts, DatasetBuilder.histories(bursts, programs, rng)):
                out[pattern][0].append(history_window(hist))
                out[pattern][1].append(normalize_burst(burst))
                out[pattern][2].append(hist)
        return out

    @staticmethod
    def build_dataset(num_workloads: int, seed: int, patterns: Optional[Sequence[str]] = None) -> BurstDataset:
        groups = DatasetBuilder.generate(num_workloads, seed, patterns)
        X = [x for g in groups.values() for x in g[0]]
        y = [t for g in groups.values() for t in g[1]]
        return BurstDataset(X, y)
