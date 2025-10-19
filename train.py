# train.py
import argparse
import os
from stable_baselines3 import PPO, A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecMonitor
from isaac_lite.env import IsaacLiteEnv

def make_env(seed, persona):
    def _init():
        env = IsaacLiteEnv(seed=seed, persona=persona)
        return env
    return _init

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--algo", choices=["ppo","a2c"], default="ppo")
    parser.add_argument("--timesteps", type=int, default=200000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--persona", choices=["survivor","explorer"], default="survivor")
    parser.add_argument("--logdir", default="runs")
    args = parser.parse_args()
    os.makedirs(args.logdir, exist_ok=True)
    env = DummyVecEnv([make_env(args.seed, args.persona)])
    env = VecMonitor(env)
    policy_kwargs = dict(net_arch=[dict(pi=[64,64], vf=[64,64])])
    if args.algo == "ppo":
        model = PPO("MlpPolicy", env, verbose=1, seed=args.seed, policy_kwargs=policy_kwargs, tensorboard_log=args.logdir)
    else:
        model = A2C("MlpPolicy", env, verbose=1, seed=args.seed, policy_kwargs=policy_kwargs)
    model.learn(total_timesteps=args.timesteps)
    model.save(f"{args.logdir}/{args.algo}_{args.persona}_seed{args.seed}")
    env.env_method("save_episode_metrics")
    print("Training complete. Model saved.")

if __name__ == "__main__":
    main()
