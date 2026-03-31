# backend/rl/reward.py

from typing import List
from backend.simulator.process import Process
from backend.simulator.core import Core
from backend.simulator.fairness import FairnessEngine


class RewardEngine:
    """
    PPO-stable reward function for CPU scheduling.

    Key improvements:
    - Strong completion incentive
    - Queue reduction pressure
    - Balanced fairness signal
    - Proper scaling (critical for PPO)
    """

    def __init__(self):

        # Tuned weights (IMPORTANT)
        self.w_completion = 10.0
        self.w_waiting = 0.05
        self.w_context = 0.2
        self.w_fairness = 1.0
        self.w_starvation = 2.0
        self.w_queue = 0.1

        self.starvation_threshold = 50
        self.prev_context_switches = 0

    # ------------------------------------------------------------

    def compute(
        self,
        completed: List[Process],
        ready_queue: List[Process],
        cores: List[Core]
    ) -> float:

        reward = 0.0

        # --------------------------------------------------------
        # 1️⃣ Strong Completion Reward (CRITICAL)
        # --------------------------------------------------------
        reward += self.w_completion * len(completed)

        # Encourage progress every step
        if len(completed) == 0:
            reward -= 0.5

        # --------------------------------------------------------
        # 2️⃣ Waiting Penalty (scaled)
        # --------------------------------------------------------
        total_waiting = sum(p.waiting_time for p in ready_queue)
        reward -= self.w_waiting * total_waiting

        # --------------------------------------------------------
        # 3️⃣ Queue Pressure (VERY IMPORTANT)
        # --------------------------------------------------------
        reward -= self.w_queue * len(ready_queue)

        # --------------------------------------------------------
        # 4️⃣ Context Switch Penalty (delta-based)
        # --------------------------------------------------------
        current_context = sum(core.context_switches for core in cores)
        delta_context = current_context - self.prev_context_switches

        reward -= self.w_context * delta_context
        self.prev_context_switches = current_context

        # --------------------------------------------------------
        # 5️⃣ Fairness (scaled down for stability)
        # --------------------------------------------------------
        all_processes = ready_queue + [
            core.current_process
            for core in cores if core.current_process
        ]

        if all_processes:
            fairness = FairnessEngine.jains_cpu_fairness(all_processes)
            reward += self.w_fairness * fairness

        # --------------------------------------------------------
        # 6️⃣ Starvation Penalty
        # --------------------------------------------------------
        starved = [
            p for p in ready_queue
            if p.waiting_time >= self.starvation_threshold
        ]

        reward -= self.w_starvation * len(starved)

        return reward

    # ------------------------------------------------------------

    def reset(self):
        self.prev_context_switches = 0