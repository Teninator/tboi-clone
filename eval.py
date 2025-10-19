import argparse, os, json, glob, pandas as pd
from stable_baselines3 import PPO
from isaac_lite.env import IsaacLiteEnv

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--persona", default="survivor")
    parser.add_argument("--episodes", type=int, default=50)
    parser.add_argument("--out", default="results.csv")
    args = parser.parse_args()
    env = IsaacLiteEnv(seed=args.seed, persona=args.persona, log_dir="eval_logs")
    model = PPO.load(args.model)
    for ep in range(args.episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, rew, done, tr, info = env.step(action)
        env.save_episode_metrics()
    files = glob.glob("eval_logs/episode_*.json")
    data = []
    for f in files:
        with open(f) as fh:
            data.append(json.load(fh))
    df = pd.DataFrame(data)
    df.to_csv(args.out, index=False)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
