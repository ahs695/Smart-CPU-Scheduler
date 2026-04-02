# backend/rl/evaluate_rl.py

import os
import torch
from stable_baselines3 import PPO

from backend.ml.lstm_model import BurstPredictorLSTM
from backend.rl.env import SchedulingEnv
from backend.hybrid.hybrid_scheduler import HybridSchedulingEnv

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
        ppo_path="models/ppo_scheduler_ppo.zip",
        hybrid_path="models/ppo_scheduler_hybrid.zip",
        lstm_path="models/lstm_model.pt",
        num_processes=30,
        num_cores=2
    ):
        self.num_processes = num_processes
        self.num_cores = num_cores
        
        # Load Baseline PPO model if available
        self.ppo_model = PPO.load(ppo_path) if os.path.exists(ppo_path) else None
        
        # Load Hybrid PPO model if available
        self.hybrid_model = PPO.load(hybrid_path) if os.path.exists(hybrid_path) else None
        
        # Backwards compatibility check for the un-suffixed hybrid model trained previously
        if not self.hybrid_model and os.path.exists("models/ppo_scheduler.zip"):
            self.hybrid_model = PPO.load("models/ppo_scheduler.zip")
            
        # Load LSTM prediction module gracefully for Hybrid logic
        self.lstm = BurstPredictorLSTM()
        if os.path.exists(lstm_path):
            checkpoint = torch.load(lstm_path, map_location="cpu")
            if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
                self.lstm.load_state_dict(checkpoint["model_state_dict"])
            else:
                self.lstm.load_state_dict(checkpoint)
        self.lstm.eval()

    # ------------------------------------------------------------

    def evaluate_rl(self, processes, mode="hybrid"):
        model = self.hybrid_model if mode == "hybrid" else self.ppo_model
        
        if model is None:
            return None

        if mode == "hybrid":
            env = HybridSchedulingEnv(
                processes=processes,
                num_cores=self.num_cores,
                max_queue_size=50,
                max_steps=100000,
                lstm_model=self.lstm,
                use_predictions=True
            )
        else:
            env = SchedulingEnv(
                processes=processes,
                num_cores=self.num_cores,
                max_queue_size=50,
                max_steps=100000
            )

        state, _ = env.reset()

        steps = 0
        safety_max = 100000
        
        preemption_events = 0

        while True:
            prev_cs = sum(c.context_switches for c in env.simulator.cores)
            
            action, _ = model.predict(state, deterministic=True)
            state, _, terminated, truncated, _ = env.step(action)
            steps += 1
            
            curr_cs = sum(c.context_switches for c in env.simulator.cores)
            if curr_cs > prev_cs:
                preemption_events += 1

            if terminated:
                break
                
            if truncated:
                print(f"⚠ [{mode.upper()}] Episode unexpectedly truncated (hit 100k limit).")
                break

            if steps >= safety_max:
                raise RuntimeError(f"[{mode.upper()}] Evaluation stuck in infinite loop.")

        sim = env.simulator
        
        completed_count = len(sim.completed_processes)
        total_processes = len(sim.processes)
        total_cs = sum(c.context_switches for c in sim.cores)
        
        title = "PPO+LSTM" if mode == "hybrid" else "PPO"
        print(f"\n--- {title} Stats ---")
        percent_completed = (completed_count / total_processes) * 100
        print(f"Processes Completed: {completed_count}/{total_processes} ({percent_completed:.2f}%)")
        print(f"Total Simulation Time: {sim.time}")
        
        percent_preemption_steps = (preemption_events / steps) * 100 if steps > 0 else 0
        print(f"Total Context Switches: {total_cs}")
        print(f"% Steps with preemption: {percent_preemption_steps:.2f}%")
        
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

        # Must copy processes structurally or regenerate workload if mutability exists natively, 
        # however MultiCoreSimulator.reset strictly re-establishes limits so passing the same original object list is functionally pure.
        org_processes = WorkloadGenerator.mixed(self.num_processes)

        results = {}

        if self.ppo_model:
            print("Evaluating Baseline PPO...")
            results["PPO"] = self.evaluate_rl(org_processes, mode="ppo")
        else:
            print("⚠ Baseline PPO model missing. Skipping (--mode ppo).")

        if self.hybrid_model:
            print("Evaluating Hybrid AI (PPO + LSTM)...")
            results["PPO+LSTM"] = self.evaluate_rl(org_processes, mode="hybrid")
        else:
            print("⚠ Hybrid AI model missing. Skipping (--mode hybrid).")


        print("\nEvaluating FCFS...")
        results["FCFS"] = self.evaluate_classical(
            FCFSScheduler(), org_processes
        )

        print("Evaluating SJF...")
        results["SJF"] = self.evaluate_classical(
            SJFScheduler(preemptive=False), org_processes
        )

        print("Evaluating RR...")
        results["RR"] = self.evaluate_classical(
            RoundRobinScheduler(quantum=2), org_processes
        )

        print("Evaluating MLFQ...")
        results["MLFQ"] = self.evaluate_classical(
            MLFQScheduler(), org_processes
        )

        self._print_results(results)

    # ------------------------------------------------------------

    def _print_results(self, results):

        print("\n================ COMPARISON ================\n")

        # Find first valid dictionary to pull keys from
        keys = []
        for v in results.values():
            if v is not None:
                keys = list(v.keys())
                break

        for algo in results:
            print(f"\n--- {algo} ---")
            if results[algo] is None:
                print("No Data Logged")
                continue
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