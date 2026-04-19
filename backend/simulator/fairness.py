from typing import List
from .process import Process


class FairnessEngine:

    @staticmethod
    def normalized_turnaround_times(processes: List[Process]):
        """
        Effective CPU rate received by each process (Burst Time / Turnaround Time).
        """
        return [p.burst_time / p.turnaround_time for p in processes if p.turnaround_time is not None and p.turnaround_time > 0]

    @staticmethod
    def jains_index(values: List[float]) -> float:

        if not values:
            return 0

        numerator = sum(values) ** 2
        denominator = len(values) * sum(v ** 2 for v in values)

        if denominator == 0:
            return 0

        return numerator / denominator

    @staticmethod
    def jains_cpu_fairness(processes: List[Process]) -> float:

        shares = FairnessEngine.normalized_turnaround_times(processes)
        return FairnessEngine.jains_index(shares)

    @staticmethod
    def waiting_time_variance(processes: List[Process]):

        waits = [p.waiting_time for p in processes]

        if not waits:
            return 0

        mean = sum(waits) / len(waits)

        return sum((w - mean) ** 2 for w in waits) / len(waits)

    @staticmethod
    def detect_starvation(processes: List[Process], threshold: int = 50):

        starved = []

        for p in processes:
            if p.waiting_time >= threshold:
                starved.append(p.pid)

        return starved