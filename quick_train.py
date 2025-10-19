import os
from stable_baselines3 import PPO
from isaac_lite.env import IsaacLiteEnv

# Config
PERSONAS = ["survivor", "explorer"]
TIMESTEPS = 10_000          # timesteps per run
RUNS_PER_PERSONA = 50       # number of independent runs per persona

for persona in PERSONAS:
    MODEL_DIR = f"logs/ppo_{persona}"
    os.makedirs(MODEL_DIR, exist_ok=True)

    for run_id in range(RUNS_PER_PERSONA):
        print(f"\n=== Training {persona} run {run_id+1}/{RUNS_PER_PERSONA} ===")

        # Initialize environment
        env = IsaacLiteEnv(persona=persona)

        # Create PPO model for vector observations
        model = PPO("MlpPolicy", env, verbose=1)

        # Train the model
        model.learn(total_timesteps=TIMESTEPS)

        # Save model
        model_path = f"{MODEL_DIR}/ppo_{persona}_run{run_id+1}_final"
        model.save(model_path)
        print(f"Model saved at: {model_path}.zip")

        # Clean up env to avoid memory issues
        env.close()
