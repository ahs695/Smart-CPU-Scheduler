import random
from typing import List, Tuple

import torch
from torch.utils.data import Dataset

from backend.simulator.process import Process
from backend.simulator.workload_generator import WorkloadGenerator


class BurstDataset(Dataset):
    """
    PyTorch dataset for CPU burst prediction.

    Each sample:
        input  -> sequence of past burst lengths
        target -> next burst length
    """

    def __init__(self, sequences: List[List[float]], targets: List[float]):
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.targets = torch.tensor(targets, dtype=torch.float32)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):

        return self.sequences[idx], self.targets[idx]


# ------------------------------------------------------------


class DatasetBuilder:
    """
    Builds datasets for LSTM burst prediction.
    """

    # ------------------------------------------------------------

    @staticmethod
    def sliding_window(
        bursts: List[int],
        sequence_length: int
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Convert burst list into sliding window dataset.
        """

        X = []
        y = []

        for i in range(len(bursts) - sequence_length):

            seq = bursts[i:i + sequence_length]
            target = bursts[i + sequence_length]

            X.append(seq)
            y.append(target)

        return X, y

    # ------------------------------------------------------------

    @staticmethod
    def generate_synthetic_sequences(
        num_sequences: int = 1000,
        seq_length: int = 5
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Generate synthetic CPU burst patterns.
        """

        sequences = []
        targets = []

        for _ in range(num_sequences):

            base = random.randint(5, 20)

            bursts = []

            for _ in range(seq_length + 1):

                noise = random.randint(-3, 3)
                burst = max(1, base + noise)

                bursts.append(burst)

            X, y = DatasetBuilder.sliding_window(bursts, seq_length)

            sequences.extend(X)
            targets.extend(y)

        return sequences, targets

    # ------------------------------------------------------------

    @staticmethod
    def from_workload_generator(
        num_processes: int = 200,
        seq_length: int = 5
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Generate dataset using workload generator bursts.
        """

        processes = WorkloadGenerator.mixed(num_processes)

        bursts = [p.burst_time for p in processes]

        return DatasetBuilder.sliding_window(bursts, seq_length)

    # ------------------------------------------------------------

    @staticmethod
    def from_simulator_traces(
        gantt_chart,
        seq_length: int = 5
    ) -> Tuple[List[List[int]], List[int]]:
        """
        Extract burst sequences from simulator execution traces.

        gantt_chart format:
            {core_id: [pid, pid, None, pid, ...]}
        """

        burst_sequences = {}

        for core in gantt_chart:

            timeline = gantt_chart[core]

            prev = None
            count = 0

            for pid in timeline:

                if pid == prev:
                    count += 1
                else:

                    if prev is not None:
                        burst_sequences.setdefault(prev, []).append(count)

                    prev = pid
                    count = 1

            if prev is not None:
                burst_sequences.setdefault(prev, []).append(count)

        bursts = []

        for pid in burst_sequences:
            bursts.extend(burst_sequences[pid])

        return DatasetBuilder.sliding_window(bursts, seq_length)

    # ------------------------------------------------------------

    @staticmethod
    def normalize(sequences, targets):
        """
        Normalize burst values for stable training.
        """

        max_val = max(max(seq) for seq in sequences)

        sequences = [[v / max_val for v in seq] for seq in sequences]
        targets = [v / max_val for v in targets]

        return sequences, targets, max_val

    # ------------------------------------------------------------

    @staticmethod
    def build_dataset(
        num_sequences: int = 1000,
        seq_length: int = 5
    ) -> BurstDataset:
        """
        Build final PyTorch dataset.
        """

        X, y = DatasetBuilder.generate_synthetic_sequences(
            num_sequences,
            seq_length
        )

        X, y, _ = DatasetBuilder.normalize(X, y)

        return BurstDataset(X, y)
    
