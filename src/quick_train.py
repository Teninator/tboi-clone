import os, time
from stable_baselines3 import PPO, A2C
from isaac_lite.env import IsaacLiteEnv

# Config
PERSONAS = ["survivor", "explorer"]
ALGOS = ["ppo", "a2c"]
TIMESTEPS = 10000
RUNS_PER_PERSONA = 3

for persona in PERSONAS:
    for algo in ALGOS:
        MODEL_DIR = f"logs/{algo}_{persona}"
        os.makedirs(MODEL_DIR, exist_ok=True)

        for run_id in range(RUNS_PER_PERSONA):
            print(f"\n=== Training {algo.upper()} for {persona} | Run {run_id+1}/{RUNS_PER_PERSONA} ===")
            env = IsaacLiteEnv(persona=persona, seed=run_id)
            model_class = PPO if algo == "ppo" else A2C
            model = model_class("MlpPolicy", env, verbose=1)
            model.learn(total_timesteps=TIMESTEPS)

            timestamp = time.strftime("%Y%m%d_%H%M%S")
            model_path = f"{MODEL_DIR}/{algo}_{persona}_run{run_id+1}_{timestamp}.zip"
            model.save(model_path)
            print(f"Saved {algo.upper()} model: {model_path}!")
            env.close()
