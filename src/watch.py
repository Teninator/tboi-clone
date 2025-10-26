import os
import pygame
import numpy as np
from stable_baselines3 import PPO, A2C
from isaac_lite.env import IsaacLiteEnv
from isaac_lite.game import ROOM_W, ROOM_H


def list_and_select_model(folder, algo_name):
    """List available models and let user pick by index."""
    if not os.path.exists(folder):
        raise FileNotFoundError(f"Model folder not found: {folder}")

    models = [f for f in os.listdir(folder) if f.endswith(".zip") and algo_name in f.lower()]
    if not models:
        raise FileNotFoundError(f"No models found in {folder}")

    models.sort()
    print(f"\n=== {algo_name.upper()} MODELS FOUND ===")
    for i, m in enumerate(models, 1):
        print(f"[{i}] {m}")

    choice = input(f"Select {algo_name.upper()} run (1-{len(models)}), or Enter for latest: ").strip()
    if choice.isdigit():
        idx = int(choice)
        selected = models[min(max(1, idx), len(models)) - 1]
    else:
        selected = models[-1]

    path = os.path.join(folder, selected)
    print(f"Loaded {algo_name.upper()} model: {path}")
    return path


def load_model(path, algo):
    """Load PPO or A2C model from disk."""
    if algo == "ppo":
        return PPO.load(path)
    elif algo == "a2c":
        return A2C.load(path)
    else:
        raise ValueError("Unknown algorithm type.")


def match_obs_shape(model, obs):
    """Ensures observation matches model’s expected input size."""
    expected_dim = model.observation_space.shape[0]
    if len(obs) > expected_dim:
        return obs[:expected_dim]
    elif len(obs) < expected_dim:
        padded = np.zeros(expected_dim)
        padded[:len(obs)] = obs
        return padded
    return obs


def main():
    print("🎮 Comparing PPO and A2C agents side by side. Press ESC to quit.")

    # Path autodetect for your project structure
    ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    LOG_DIR = os.path.join(ROOT_DIR, "logs")

    ppo_dir = os.path.join(LOG_DIR, "ppo_explorer")
    a2c_dir = os.path.join(LOG_DIR, "a2c_explorer")

    # Load models interactively
    ppo_path = list_and_select_model(ppo_dir, "ppo")
    a2c_path = list_and_select_model(a2c_dir, "a2c")

    model_ppo = load_model(ppo_path, "ppo")
    model_a2c = load_model(a2c_path, "a2c")

    # Create environments
    env_left = IsaacLiteEnv(persona='explorer')
    env_right = IsaacLiteEnv(persona='explorer')

    obs_left, _ = env_left.reset()
    obs_right, _ = env_right.reset()

    # Init display
    pygame.init()
    screen_width = (ROOM_W + 150) * 2
    screen_height = ROOM_H
    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption("PPO vs A2C Comparison")

    clock = pygame.time.Clock()
    running = True

    while running:
        # Handle input events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        # Match shapes automatically before prediction
        obs_left = match_obs_shape(model_ppo, obs_left)
        obs_right = match_obs_shape(model_a2c, obs_right)

        # Predict actions
        action_ppo, _ = model_ppo.predict(obs_left, deterministic=True)
        action_a2c, _ = model_a2c.predict(obs_right, deterministic=True)

        # Step both environments
        obs_left, _, done_left, _, _ = env_left.step(int(action_ppo))
        obs_right, _, done_right, _, _ = env_right.step(int(action_a2c))

        # Render frames
        frame_left = env_left.render()
        frame_right = env_right.render()

        surf_left = pygame.surfarray.make_surface(np.transpose(frame_left, (1, 0, 2)))
        surf_right = pygame.surfarray.make_surface(np.transpose(frame_right, (1, 0, 2)))

        screen.blit(surf_left, (0, 0))
        screen.blit(surf_right, (ROOM_W + 150, 0))

        pygame.display.flip()
        clock.tick(30)

        # Auto-reset on episode end
        if done_left:
            obs_left, _ = env_left.reset()
        if done_right:
            obs_right, _ = env_right.reset()

    pygame.quit()


if __name__ == "__main__":
    main()
