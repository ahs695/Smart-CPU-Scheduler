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