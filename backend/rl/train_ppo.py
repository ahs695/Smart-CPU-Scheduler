import os

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from backend.rl.env import SchedulingEnv
from backend.simulator.workload_generator import WorkloadGenerator


class PPOTrainer:
    """
    Trains PPO agent for CPU scheduling.
    """

    def __init__(
        self,
        num_processes: int = 30,
        num_cores: int = 2,
        total_timesteps: int = 150_000,
        model_path: str = "models/ppo_scheduler"
    ):

        self.num_processes = num_processes
        self.num_cores = num_cores
        self.total_timesteps = total_timesteps
        self.model_path = model_path

        os.makedirs("models", exist_ok=True)

        # --------------------------------------------------------
        # Create environment
        # --------------------------------------------------------
        # We generate a large enough pool for diverse episodes
        processes = WorkloadGenerator.mixed(num_processes)

        def make_env():
            return SchedulingEnv(
                processes=processes,
                num_cores=num_cores,
                max_queue_size=50,
                max_steps=5000
            )

        self.env = DummyVecEnv([make_env])

        # --------------------------------------------------------
        # PPO Model
        # --------------------------------------------------------
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            learning_rate=3e-4,     # fixed parameter requirement
            n_steps=2048,           # fixed parameter requirement
            batch_size=64,          # fixed parameter requirement
            gamma=0.99,             # fixed parameter requirement
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01           # reduced entropy since task is quite deterministic
        )

    # ------------------------------------------------------------

    def train(self):

        print("\nStarting PPO training...\n")

        self.model.learn(total_timesteps=self.total_timesteps)

        self.model.save(self.model_path)

        print(f"\nModel saved to: {self.model_path}.zip")


# ------------------------------------------------------------


def main():

    trainer = PPOTrainer(
        num_processes=30,
        num_cores=2,
        total_timesteps=150_000
    )

    trainer.train()


if __name__ == "__main__":
    main()