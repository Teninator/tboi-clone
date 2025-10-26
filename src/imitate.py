import json
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from isaac_lite.env import IsaacLiteEnv


def train_from_human_data(data_path="logs/human_sessions/", persona='survivor'):
    # Auto-detect session file
    if os.path.isdir(data_path):
        json_files = [f for f in os.listdir(data_path) if f.endswith(".json")]
        if not json_files:
            raise FileNotFoundError(f"No JSON files found in {data_path}")
        data_path = os.path.join(data_path, sorted(json_files)[-1])

    print(f"📁 Using human data file: {data_path}")

    # Load human session data
    with open(data_path, "r") as f:
        data = json.load(f)

    if not data:
        raise ValueError("❌ No data found in human session file. Did you record gameplay first?")

    obs = np.array([d["obs"] for d in data], dtype=np.float32)
    actions = np.array([d["action"] for d in data], dtype=np.int64)

    # Create environment
    seed = 42
    env = make_vec_env(lambda: IsaacLiteEnv(persona=persona), n_envs=1, seed=seed)

    # Initialize PPO
    model = PPO("MlpPolicy", env, verbose=1, device="cpu")

    # Behavioral cloning pretraining
    optimizer = model.policy.optimizer
    optimizer.zero_grad()

    total_loss = 0
    for o, a in zip(obs, actions):
        o_tensor = torch.tensor([o], dtype=torch.float32)
        a_tensor = torch.tensor([a], dtype=torch.long)
        loss, _, _ = model.policy.evaluate_actions(o_tensor, a_tensor)
        loss.mean().backward()
        total_loss += loss.item()

    optimizer.step()

    model_path = f"logs/ppo_bc_pretrained_{persona}.zip"
    model.save(model_path)
    print(f"Saved imitation-trained model to: {model_path}")
    print(f"Average imitation loss: {total_loss / len(obs):.4f}")


if __name__ == "__main__":
    try:
        train_from_human_data("logs/human_sessions/", persona='survivor')
    except Exception as e:
        print(f"Error during imitation training: {e}")
        print("Make sure you have a human gameplay recording from solo.py in 'logs/human_sessions/'")
