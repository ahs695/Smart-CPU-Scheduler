# backend/rl/evaluate_rl.py

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
        self.model = PPO.load(model_path)
        self.num_processes = num_processes
        self.num_cores = num_cores

    # ------------------------------------------------------------

    def evaluate_rl(self, processes):

        env = SchedulingEnv(
            processes=processes,
            num_cores=self.num_cores
        )

        state, _ = env.reset()

        steps = 0
        max_steps = 2000   # safety cap

        while True:

            action, _ = self.model.predict(state, deterministic=True)

            state, _, terminated, truncated, _ = env.step(action)

            steps += 1

            # ✅ Proper stopping condition
            if terminated:
                break

            # ⚠️ If truncated, KEEP GOING (important fix)
            if truncated:
                continue

            # safety break
            if steps >= max_steps:
                print("⚠ Safety break triggered")
                break

        sim = env.simulator

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

        print("\nGenerating workload...\n")

        processes = WorkloadGenerator.mixed(self.num_processes)

        results = {}

        print("Evaluating PPO...")
        results["PPO"] = self.evaluate_rl(processes)

        print("Evaluating FCFS...")
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
    evaluator = RLEvaluator()
    evaluator.run()


if __name__ == "__main__":
    main()