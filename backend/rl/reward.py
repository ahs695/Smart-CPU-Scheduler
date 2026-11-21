# backend/rl/reward.py

class RewardEngine:
    """
    Per-tick reward aligned exactly with the evaluation metrics.

        r_t = -(jobs in the system during tick t) / scale
              - switch_cost * (context switches this tick)

    Summed over an episode, the first term is -(sum of turnaround times) / scale
    (Little's law), so maximising return minimises average turnaround — and
    average waiting time, which is turnaround minus the fixed burst. The second
    term charges for each context switch, which is not free on a real CPU.

    Deadline terms (zero for workloads without deadlines):
        - miss_penalty per job whose deadline passes unfinished this tick
        - lateness_cost per tick for every job already past its deadline;
          summed over an episode this is lateness_cost × total lateness.

    There is deliberately no squashing (the old tanh flattened every step with
    a completion to ~1.0); reward scale is handled by VecNormalize in training.
    """

    def __init__(self, scale: float = 10.0, switch_cost: float = 0.05,
                 miss_penalty: float = 1.0, lateness_cost: float = 0.1):
        self.scale = scale
        self.switch_cost = switch_cost
        self.miss_penalty = miss_penalty
        self.lateness_cost = lateness_cost

    def compute(self, jobs_in_system: int, context_switches: int,
                new_misses: int = 0, overdue_jobs: int = 0) -> float:
        return (-jobs_in_system / self.scale
                - self.switch_cost * context_switches
                - self.miss_penalty * new_misses
                - self.lateness_cost * overdue_jobs)
