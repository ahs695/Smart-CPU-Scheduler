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

    There is deliberately no squashing (the old tanh flattened every step with
    a completion to ~1.0); reward scale is handled by VecNormalize in training.
    """

    def __init__(self, scale: float = 10.0, switch_cost: float = 0.05):
        self.scale = scale
        self.switch_cost = switch_cost

    def compute(self, jobs_in_system: int, context_switches: int) -> float:
        return -jobs_in_system / self.scale - self.switch_cost * context_switches
