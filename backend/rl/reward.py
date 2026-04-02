# backend/rl/reward.py

import numpy as np
from typing import List, Tuple
from backend.simulator.process import Process
from backend.simulator.core import Core


class RewardEngine:
    """
    Hybrid AI Delta-Based Reward Engine.
    
    Teaches dynamic preemptive scheduling by measuring delta waiting time
    and explicitly pushing the agent to favor shorter LSTM-predicted bursts.
    """

    def __init__(self):
        # Weights
        self.w_completion = 5.0       
        self.w_delta_wait = 0.05      
        self.w_idle_penalty = 0.05    
        
        # Context switch dynamic weights
        self.w_context_good = 0.005    # Switched to lower-predicted job
        self.w_context_bad = 0.05      # Unnecessary/worse switch
        
        self.w_starvation = 0.02      
        self.w_pred_burst = 0.05       # Penalty constant for running long jobs

        self.starvation_threshold = 100
        self.prev_context_switches = 0
        self.prev_running_burst = 0.0

    # ------------------------------------------------------------

    def compute(
        self,
        completed: List[Process],
        ready_queue: List[Process],
        cores: List[Core],
        prev_wait: float,
        curr_wait: float,
        selected_preds: List[Tuple[int, float]]
    ) -> float:

        reward = 0.0

        # 1️⃣ Throughput Gain
        reward += self.w_completion * len(completed)

        # 2️⃣ Delta Waiting Time
        delta_wait = curr_wait - prev_wait
        reward -= self.w_delta_wait * delta_wait

        # 3️⃣ Idle Core Penalty
        idle_cores = sum(1 for core in cores if core.current_process is None)
        if ready_queue:
            reward -= self.w_idle_penalty * idle_cores

        # 4️⃣ Burst Prediction Penalty (Drive SJF Behavior)
        # Directly penalize the agent for holding long jobs on the cores
        curr_running_burst = 0.0
        if selected_preds:
            for pid, pred_burst in selected_preds:
                reward -= self.w_pred_burst * pred_burst
            curr_running_burst = sum(pred for _, pred in selected_preds) / len(selected_preds)

        # 5️⃣ Intelligent Context Switch Penalty
        current_context = sum(core.context_switches for core in cores)
        delta_context = current_context - self.prev_context_switches
        
        if delta_context > 0:
            if curr_running_burst < self.prev_running_burst:
                # Preempted for a statistically SHORTER job! Good dynamic behavior.
                reward -= self.w_context_good * delta_context
            else:
                # Preempted for a WORSE or identical job. High penalty to prevent thrashing.
                reward -= self.w_context_bad * delta_context
                
        self.prev_context_switches = current_context
        self.prev_running_burst = curr_running_burst

        # 6️⃣ Starvation Penalty
        starved_count = sum(1 for p in ready_queue if p.waiting_time >= self.starvation_threshold)
        reward -= self.w_starvation * starved_count

        normalized_reward = np.tanh(reward)

        return float(normalized_reward)

    # ------------------------------------------------------------

    def reset(self):
        self.prev_context_switches = 0
        self.prev_running_burst = 0.0