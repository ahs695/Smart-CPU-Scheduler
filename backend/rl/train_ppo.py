# backend/rl/train_ppo.py

import os
import torch
import numpy as np
import argparse

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import BaseCallback

from backend.rl.env import SchedulingEnv
from backend.hybrid.hybrid_scheduler import HybridSchedulingEnv
from backend.simulator.workload_generator import WorkloadGenerator
from backend.ml.lstm_model import BurstPredictorLSTM


class DebugMetricsCallback(BaseCallback):
    """
    Tracks and logs custom architecture metrics from the environment infos.
    Detects presence of hybrid keys before processing dynamically.
    """
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.avg_predicted_bursts = []
        self.shortest_selections = []

    def _on_step(self) -> bool:
        for info in self.locals.get("infos", []):
            if "avg_predicted_burst" in info:
                self.avg_predicted_bursts.append(info["avg_predicted_burst"])
            if "shortest_selected" in info:
                self.shortest_selections.append(info["shortest_selected"])
        return True

    def _on_rollout_end(self):
        if self.avg_predicted_bursts:
            avg_burst = np.mean(self.avg_predicted_bursts)
            self.logger.record("custom/avg_predicted_burst", avg_burst)
            self.avg_predicted_bursts = []
        
        if self.shortest_selections:
            avg_shortest = np.mean(self.shortest_selections) * 100
            self.logger.record("custom/shortest_selected_pct", avg_shortest)
            self.shortest_selections = []


class PPOTrainer:
    """
    Trains CPU scheduling models via dynamic config architectures.
    Supports either purely mathematical 'ppo' bounds or 'hybrid' AI variants.
    """

    def __init__(
        self,
        mode: str = "hybrid",
        num_processes: int = 30,
        num_cores: int = 2,
        total_timesteps: int = 200_000,
        model_path: str = "models/ppo_scheduler"
    ):

        self.mode = mode.lower()
        self.num_processes = num_processes
        self.num_cores = num_cores
        self.total_timesteps = total_timesteps
        
        # Suffix the path by mode (ppo -> ppo_scheduler, hybrid -> ppo_scheduler_hybrid)
        if self.mode == "ppo":
            self.model_path = model_path
        else:
            self.model_path = f"{model_path}_{self.mode}"

        os.makedirs("models", exist_ok=True)
        
        # Load LSTM Component exclusively for Hybrid architectures
        self.lstm = None
        if self.mode == "hybrid":
            self.lstm = BurstPredictorLSTM()
            lstm_path = "models/lstm_model.pt"
            if os.path.exists(lstm_path):
                checkpoint = torch.load(lstm_path, map_location="cpu")
                if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                    self.lstm.load_state_dict(checkpoint["model_state_dict"])
                else:
                    self.lstm.load_state_dict(checkpoint)
            self.lstm.eval()

        processes = WorkloadGenerator.mixed(num_processes)

        def make_env():
            if self.mode == "hybrid":
                return HybridSchedulingEnv(
                    processes=processes,
                    num_cores=num_cores,
                    max_queue_size=50,
                    max_steps=15000,
                    lstm_model=self.lstm,
                    use_predictions=True
                )
            else:
                return SchedulingEnv(
                    processes=processes,
                    num_cores=num_cores,
                    max_queue_size=50,
                    max_steps=15000
                )

        self.env = DummyVecEnv([make_env])


        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            gamma=0.995,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.005 if self.mode == "hybrid" else 0.05
        )
        
        self.callback = DebugMetricsCallback()

    # ------------------------------------------------------------

    def train(self):
        print(f"\nStarting {self.mode.upper()} Agent Training Pipeline...\n")
        self.model.learn(total_timesteps=self.total_timesteps, callback=self.callback)
        self.model.save(self.model_path)
        print(f"\nModel strictly saved successfully to: {self.model_path}.zip")


# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Multi-Core Scheduler PPO Trainer")
    parser.add_argument("--mode", type=str, default="hybrid", choices=["ppo", "hybrid"],
                        help="Train purely algorithmic 'ppo' or advanced predictive 'hybrid' mappings.")
    parser.add_argument("--timesteps", type=int, default=200_000, help="Total execution limits")
    parser.add_argument("--cores", type=int, default=2, help="Number of CPU cores to simulate")
    args = parser.parse_args()

    trainer = PPOTrainer(
        mode=args.mode,
        num_processes=30,
        num_cores=args.cores,
        total_timesteps=args.timesteps
    )

    trainer.train()

if __name__ == "__main__":
    main()