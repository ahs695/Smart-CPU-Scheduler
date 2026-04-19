# backend/simulator/metrics.py

from typing import List, Dict
from .process import Process
from .core import Core


class MetricsEngine:
    """
    Research-grade metrics engine for CPU scheduling experiments.

    Robust to:
    - Incomplete simulations
    - RL instability
    - Missing values
    """

    # ------------------------------------------------------------

    @staticmethod
    def compute_waiting_times(processes: List[Process]) -> List[int]:
        return [
            p.waiting_time
            for p in processes
            if p.waiting_time is not None
        ]

    # ------------------------------------------------------------

    @staticmethod
    def compute_turnaround_times(processes: List[Process]) -> List[int]:
        return [
            p.turnaround_time
            for p in processes
            if p.turnaround_time is not None
        ]

    # ------------------------------------------------------------

    @staticmethod
    def compute_response_times(processes: List[Process]) -> List[int]:
        return [
            p.response_time
            for p in processes
            if p.response_time is not None
        ]

    # ------------------------------------------------------------

    @staticmethod
    def average(values: List[float]) -> float:
        """
        Safe average that ignores None values.
        """
        values = [v for v in values if v is not None]

        if not values:
            return 0.0

        return sum(values) / len(values)

    # ------------------------------------------------------------

    @staticmethod
    def throughput(processes: List[Process], total_time: int) -> float:
        """
        Only count completed processes.
        """
        if total_time == 0:
            return 0.0

        completed = [
            p for p in processes
            if p.turnaround_time is not None
        ]

        return len(completed) / total_time

    # ------------------------------------------------------------

    @staticmethod
    def cpu_utilization(cores: List[Core], total_time: int) -> float:

        if total_time == 0:
            return 0.0

        total_busy = sum(core.busy_time for core in cores)
        total_capacity = len(cores) * total_time

        return total_busy / total_capacity

    # ------------------------------------------------------------

    @staticmethod
    def context_switches(cores: List[Core]) -> int:
        return sum(core.context_switches for core in cores)

    # ------------------------------------------------------------

    @staticmethod
    def jains_fairness_index(processes: List[Process]) -> float:
        """
        Jain’s fairness index based on normalized turnaround time (Burst Time / Turnaround Time).
        """

        rates = []
        for p in processes:
            if p.turnaround_time is not None and p.turnaround_time > 0:
                # Effective CPU rate: what fraction of the turnaround time was spent actually running
                rates.append(p.burst_time / p.turnaround_time)

        if not rates:
            return 0.0

        numerator = sum(rates) ** 2
        denominator = len(rates) * sum(x ** 2 for x in rates)

        if denominator == 0:
            return 0.0

        return numerator / denominator

    # ------------------------------------------------------------

    @staticmethod
    def summarize(
        processes: List[Process],
        cores: List[Core],
        total_time: int
    ) -> Dict:

        # Detect incomplete processes
        incomplete = [
            p for p in processes
            if p.turnaround_time is None
        ]

        if incomplete:
            print(f"⚠ Warning: {len(incomplete)} processes did not complete")

        waiting = MetricsEngine.compute_waiting_times(processes)
        turnaround = MetricsEngine.compute_turnaround_times(processes)
        response = MetricsEngine.compute_response_times(processes)

        return {
            "avg_waiting_time": MetricsEngine.average(waiting),
            "avg_turnaround_time": MetricsEngine.average(turnaround),
            "avg_response_time": MetricsEngine.average(response),
            "throughput": MetricsEngine.throughput(processes, total_time),
            "cpu_utilization": MetricsEngine.cpu_utilization(cores, total_time),
            "context_switches": MetricsEngine.context_switches(cores),
            "fairness_index": MetricsEngine.jains_fairness_index(processes),
            "completed_processes": len(turnaround),
            "total_processes": len(processes),
            "total_time": total_time,
        }