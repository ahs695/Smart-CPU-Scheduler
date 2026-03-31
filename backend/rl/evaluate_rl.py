# backend/rl/evaluate_rl.py

import os
from stable_baselines3 import PPO

from backend.rl.env import SchedulingEnv
from backend.simulator.workload_generator import WorkloadGenerator
from backend.simulator.multi_core_simulator import MultiCoreSimulator
from backend.simulator.metrics import MetricsEngine

# Classical schedulers
from backend.simulator.traditional.fcfs import FCFSScheduler
from backend.simulator.traditional.sjf import SJFScheduler
from backend.simulator.traditional.round_robin import RoundRobinScheduler
from backend.simulator.traditional.mlfq import MLFQScheduler


class RLEvaluator:

    def __init__(
        self,
        model_path="models/ppo_scheduler.zip",
        num_processes=30,
        num_cores=2
    ):
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model {model_path} not found. Train first.")
            
        self.model = PPO.load(model_path)
        self.num_processes = num_processes
        self.num_cores = num_cores

    # ------------------------------------------------------------

    def evaluate_rl(self, processes):

        # Crucial to set max_steps extremely high so simulation never truncates early
        env = SchedulingEnv(
            processes=processes,
            num_cores=self.num_cores,
            max_queue_size=50,
            max_steps=100000 
        )

        state, _ = env.reset()

        steps = 0
        safety_max = 100000

        while True:
            action, _ = self.model.predict(state, deterministic=True)

            state, _, terminated, truncated, _ = env.step(action)
            steps += 1

            if terminated:
                break
                
            if truncated:
                print("⚠ Episode unexpectedly truncated (hit 100k limit).")
                break

            if steps >= safety_max:
                raise RuntimeError("Evaluation stuck in infinite loop.")

        sim = env.simulator
        
        # Explicit Error checking & Analytics logic
        completed_count = len(sim.completed_processes)
        total_processes = len(sim.processes)
        
        print("\n--- PPO Completion Stats ---")
        percent_completed = (completed_count / total_processes) * 100
        print(f"Processes Completed: {completed_count}/{total_processes} ({percent_completed:.2f}%)")
        print(f"Total Simulation Time: {sim.time}")
        
        if completed_count < total_processes:
            raise RuntimeError(f"🚨 FAILED! Only {completed_count}/{total_processes} processes completed!")

        metrics = MetricsEngine.summarize(
            processes=sim.processes,
            cores=sim.cores,
            total_time=sim.time
        )

        return metrics

    # ------------------------------------------------------------

    def evaluate_classical(self, scheduler, processes):

        sim = MultiCoreSimulator(
            processes=processes,
            scheduler=scheduler,
            num_cores=self.num_cores
        )

        results = sim.run()

        metrics = MetricsEngine.summarize(
            processes=results["processes"],
            cores=results["cores"],
            total_time=results["total_time"]
        )

        return metrics

    # ------------------------------------------------------------

    def run(self):

        print("\nGenerating evaluation workload...\n")

        processes = WorkloadGenerator.mixed(self.num_processes)

        results = {}

        print("Evaluating PPO...")
        results["PPO"] = self.evaluate_rl(processes)

        print("\nEvaluating FCFS...")
        results["FCFS"] = self.evaluate_classical(
            FCFSScheduler(), processes
        )

        print("Evaluating SJF...")
        results["SJF"] = self.evaluate_classical(
            SJFScheduler(preemptive=False), processes
        )

        print("Evaluating RR...")
        results["RR"] = self.evaluate_classical(
            RoundRobinScheduler(quantum=2), processes
        )

        print("Evaluating MLFQ...")
        results["MLFQ"] = self.evaluate_classical(
            MLFQScheduler(), processes
        )

        self._print_results(results)

    # ------------------------------------------------------------

    def _print_results(self, results):

        print("\n================ COMPARISON ================\n")

        keys = list(next(iter(results.values())).keys())

        for algo in results:
            print(f"\n--- {algo} ---")
            for k in keys:
                print(f"{k}: {results[algo][k]:.4f}")


# ------------------------------------------------------------


def main():
    try:
        evaluator = RLEvaluator()
        evaluator.run()
    except Exception as e:
        print(f"\n❌ Evaluation Failed: {str(e)}")


if __name__ == "__main__":
    main()