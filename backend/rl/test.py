from backend.rl.env import SchedulingEnv
from backend.simulator.workload_generator import WorkloadGenerator

processes = WorkloadGenerator.mixed(10)

env = SchedulingEnv(processes)

state, _ = env.reset()

for _ in range(20):
    action = env.action_space.sample()
    state, reward, done, _, _ = env.step(action)

    if done:
        break

print("Env test passed")