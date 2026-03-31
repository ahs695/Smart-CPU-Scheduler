# backend/rl/reward.py

import numpy as np
from typing import List
from backend.simulator.process import Process
from backend.simulator.core import Core


class RewardEngine:
    """
    Fixed PPO Dense Reward Engine.
    
    Provides highly granular, normalized steps to guide the agent
    swiftly towards maximizing throughput and minimizing wait times.
    """

    def __init__(self):

        # Finely tuned weights to keep total sum within [-1, 1] usually
        self.w_completion = 1.0       # throughput gain
        self.w_wait_penalty = 0.001   # per process per step
        self.w_idle_penalty = 0.05    # per idle core when queue is not empty
        self.w_context = 0.01         # per context switch
        self.w_starvation = 0.02      # per starved process per step

        self.starvation_threshold = 100
        self.prev_context_switches = 0

    # ------------------------------------------------------------

    def compute(
        self,
        completed: List[Process],
        ready_queue: List[Process],
        cores: List[Core]
    ) -> float:

        reward = 0.0

        # 1️⃣ Throughput Gain (Process Completion)
        reward += self.w_completion * len(completed)

        # 2️⃣ Waiting Time Penalty (Dense penalty per step)
        # Every step a process sits in the queue, we bleed a tiny bit of reward
        reward -= self.w_wait_penalty * len(ready_queue)

        # 3️⃣ Idle Core Penalty
        # CRITICAL: If there is work to do, but cores are idle, huge penalty.
        idle_cores = sum(1 for core in cores if core.current_process is None)
        if len(ready_queue) > 0:
            reward -= self.w_idle_penalty * idle_cores

        # 4️⃣ Context Switch Penalty (Delta)
        current_context = sum(core.context_switches for core in cores)
        delta_context = current_context - self.prev_context_switches
        
        reward -= self.w_context * delta_context
        self.prev_context_switches = current_context

        # 5️⃣ Starvation Penalty
        starved_count = sum(1 for p in ready_queue if p.waiting_time >= self.starvation_threshold)
        reward -= self.w_starvation * starved_count

        # Bound the reward to prevent exploding gradients (~[-1.0, 1.0])
        # Tanh is a popular normalization trick for dense PPO rewards
        normalized_reward = np.tanh(reward)

        return float(normalized_reward)

    # ------------------------------------------------------------

    def reset(self):
        self.prev_context_switches = 0