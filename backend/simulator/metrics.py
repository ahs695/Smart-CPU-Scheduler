# backend/simulator/metrics.py

from typing import List, Dict
from statistics import mean
from .process import Process
from .core import Core


class MetricsEngine:
    """
    Research-grade metrics engine for CPU scheduling experiments.
    Computes classical performance metrics and fairness measures.
    """

    # ------------------------------------------------------------

    @staticmethod
    def compute_waiting_times(processes: List[Process]) -> List[int]:
        return [p.waiting_time for p in processes]

    # ------------------------------------------------------------

    @staticmethod
    def compute_turnaround_times(processes: List[Process]) -> List[int]:
        return [p.turnaround_time for p in processes]

    # ------------------------------------------------------------

    @staticmethod
    def compute_response_times(processes: List[Process]) -> List[int]:
        return [p.response_time for p in processes]

    # ------------------------------------------------------------

    @staticmethod
    def average(values: List[float]) -> float:
        if not values:
            return 0.0
        return mean(values)

    # ------------------------------------------------------------

    @staticmethod
    def throughput(processes: List[Process], total_time: int) -> float:
        if total_time == 0:
            return 0
        return len(processes) / total_time

    # ------------------------------------------------------------

    @staticmethod
    def cpu_utilization(cores: List[Core], total_time: int) -> float:

        if total_time == 0:
            return 0

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
        Uses CPU time received by each process as resource share.
        """

        cpu_shares = [p.burst_time - p.remaining_time for p in processes]

        numerator = sum(cpu_shares) ** 2
        denominator = len(cpu_shares) * sum(x ** 2 for x in cpu_shares)

        if denominator == 0:
            return 0

        return numerator / denominator

    # ------------------------------------------------------------

    @staticmethod
    def summarize(
        processes: List[Process],
        cores: List[Core],
        total_time: int
    ) -> Dict:

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
            "total_processes": len(processes),
            "total_time": total_time,
        }
        
