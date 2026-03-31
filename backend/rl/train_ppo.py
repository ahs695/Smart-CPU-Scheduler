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
        num_processes: int = 20,
        num_cores: int = 2,
        total_timesteps: int = 500_000,
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
        processes = WorkloadGenerator.mixed(num_processes)

        def make_env():
            return SchedulingEnv(
                processes=processes,
                num_cores=num_cores
            )

        self.env = DummyVecEnv([make_env])

        # --------------------------------------------------------
        # PPO Model
        # --------------------------------------------------------
        self.model = PPO(
            policy="MlpPolicy",
            env=self.env,
            verbose=1,
            learning_rate=3e-4,
            n_steps=1024,
            batch_size=64,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.05
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
        total_timesteps=700_000
    )

    trainer.train()


if __name__ == "__main__":
    main()