# backend/hybrid/hybrid_scheduler.py

import numpy as np
import torch
from gymnasium import spaces

from backend.rl.env import SchedulingEnv
from backend.ml.lstm_model import BurstPredictorLSTM


class HybridSchedulingEnv(SchedulingEnv):
    """
    Hybrid AI CPU Scheduler.
    Extends the baseline PPO environment to intelligently ingest
    LSTM burst-predictions natively mapping proactive SJF logic.
    """

    def __init__(
        self,
        processes,
        num_cores: int = 2,
        max_queue_size: int = 50,
        max_steps: int = 15000,
        lstm_model=None,
        use_predictions: bool = True
    ):
        # 1. Initialize base class
        super().__init__(
            processes=processes,
            num_cores=num_cores,
            max_queue_size=max_queue_size,
            max_steps=max_steps
        )
        
        # 2. Hybrid specifics
        self.use_predictions = use_predictions
        self.lstm_model = lstm_model
        
        if self.use_predictions and self.lstm_model is None:
            self.lstm_model = BurstPredictorLSTM()
            self.lstm_model.eval()

        # 3. Augment state observation vector
        # Previous: max_choices * 3 + global
        # Hybrid: max_choices * 4 + global  (adding predicted_burst per process)
        obs_size = (
            self.max_choices * 4
            + 1  # queue len
            + self.num_cores
            + 1  # time progress
        )

        self.observation_space = spaces.Box(
            low=0,
            high=1,
            shape=(obs_size,),
            dtype=np.float32
        )

        self.process_predictions = {}
        self.MAX_BURST = 50.0

        # Run precomputation immediately
        self._precompute_lstm_predictions()

    # ------------------------------------------------------------
    
    def _precompute_lstm_predictions(self):
        """
        Calculates LSTM burst predictions for all processes perfectly synchronously.
        Uses the last 5 processes' actual burst times as input sequence window.
        """
        self.process_predictions = {}
        
        # Sort by arrival
        sorted_procs = sorted(self.original_processes, key=lambda p: p.arrival_time)
        burst_history = []
        
        for p in sorted_procs:
            if not self.use_predictions:
                # If toggled off, feed a naive 0 baseline
                self.process_predictions[p.pid] = 0.0
                continue
                
            # Pad history if length < 5
            seq_length = 5
            if len(burst_history) < seq_length:
                seq = [0.0] * (seq_length - len(burst_history)) + burst_history
            else:
                seq = burst_history[-seq_length:]
                
            # Predict precisely isolated via torch.no_grad()
            with torch.no_grad():
                X = torch.tensor([seq], dtype=torch.float32)
                pred = self.lstm_model(X).item()
                
            pred_clamped = min(max(pred, 0.0), 1.0)
            self.process_predictions[p.pid] = pred_clamped
            
            # Add ACTUAL burst to history for NEXT process organically
            burst_history.append(min(p.burst_time / self.MAX_BURST, 1.0))

    # ------------------------------------------------------------

    def reset(self, seed=None, options=None):
        # Allow super logic to handle simulator/reward resets
        state, info = super().reset(seed=seed, options=options)

        # Ensure predictions are flushed if structures rebuild
        if not self.process_predictions:
            self._precompute_lstm_predictions()
            
        # Reclaim the strictly hybridized state structure
        return self._get_state(), info

    # ------------------------------------------------------------
    
    def step(self, action):
        """
        Intercepts step method strictly to inject localized LSTM metadata correctly.
        """
        self.current_action = action
        self.invalid_action_penalty = 0.0
        
        ready_processes = self.simulator.ready_queue + [c.current_process for c in self.simulator.cores if c.current_process]
        self.prev_total_waiting = sum(p.waiting_time for p in ready_processes) if ready_processes else 0.0

        self.steps += 1

        completed, sim_all_done = self.simulator.step()

        new_ready = self.simulator.ready_queue + [c.current_process for c in self.simulator.cores if c.current_process]
        curr_total_waiting = sum(p.waiting_time for p in new_ready) if new_ready else 0.0
        
        state = self._get_state()
        
        # Calculate selected predictions expressly for the reward engine integration
        selected_preds = []
        for c in self.simulator.cores:
            if c.current_process:
                pred = self.process_predictions.get(c.current_process.pid, 0.0)
                selected_preds.append((c.current_process.pid, pred))

        step_reward = self.reward_engine.compute(
            completed=completed,
            ready_queue=new_ready,
            cores=self.simulator.cores,
            prev_wait=self.prev_total_waiting,
            curr_wait=curr_total_waiting,
            selected_preds=selected_preds
        )

        reward = step_reward + self.invalid_action_penalty

        terminated = sim_all_done
        truncated = self.steps >= self.max_steps
        
        if truncated and not terminated:
            incomplete = [
                p for p in self.simulator.processes
                if p.turnaround_time is None
            ]
            reward -= 2.0 * len(incomplete)

        # Calculate Debug metrics dynamically inside trajectory
        info = {}
        if selected_preds:
            avg_pred = sum(p[1] for p in selected_preds) / len(selected_preds)
            info["avg_predicted_burst"] = avg_pred
            
            if new_ready:
                pool_preds = [self.process_predictions.get(p.pid, 1.0) for p in new_ready]
                min_pred = min(pool_preds)
                chose_shortest = any(abs(p[1] - min_pred) < 1e-3 for p in selected_preds)
                info["shortest_selected"] = 1.0 if chose_shortest else 0.0
            else:
                info["shortest_selected"] = 0.0

        return state, reward, terminated, truncated, info

    # ------------------------------------------------------------

    def _get_state(self):
        """
        Augmented state compiler integrating the specific LSTM scalar seamlessly.
        """
        pool = list(self.simulator.ready_queue)
        for c in self.simulator.cores:
            if c.current_process and c.current_process not in pool:
                pool.append(c.current_process)

        process_features = []
        MAX_WAIT = 1000.0
        MAX_TIME = float(self.max_steps)

        for i in range(self.max_choices):
            if i < len(pool):
                p = pool[i]
                rem = min(p.remaining_time / self.MAX_BURST, 1.0)
                wait = min(p.waiting_time / MAX_WAIT, 1.0)
                pred = self.process_predictions.get(p.pid, 0.0)
                arr = min(p.arrival_time / MAX_TIME, 1.0)
                process_features.extend([rem, wait, pred, arr])
            else:
                process_features.extend([0.0, 0.0, 0.0, 0.0])

        q_len = min(len(self.simulator.ready_queue), self.max_queue_size) / self.max_queue_size
        core_busy = [1.0 if core.current_process else 0.0 for core in self.simulator.cores]
        time_norm = min(self.simulator.time / self.max_steps, 1.0)

        state_list = process_features + [q_len] + core_busy + [time_norm]
        state = np.array(state_list, dtype=np.float32)

        return state
