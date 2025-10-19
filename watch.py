import os
import glob
import time
import random
import pygame
from stable_baselines3 import PPO
from isaac_lite.env import IsaacLiteEnv

# --- Setup ---
pygame.init()
fps = 30
frame_delay = 1 / fps

# Function to load the latest model for a persona
def load_latest_model(persona):
    folder = f"logs/ppo_{persona}"
    files = sorted(glob.glob(f"{folder}/ppo_{persona}_run*_final.zip"))
    if not files:
        raise FileNotFoundError(f"No trained models found for persona '{persona}' in {folder}")
    latest = random.choice(files)
    print(f"✅ Loaded latest model for {persona}: {latest}")
    return PPO.load(latest)

# Start with explorer
current_persona = "explorer"
model = load_latest_model(current_persona)
env = IsaacLiteEnv(persona=current_persona)

obs, _ = env.reset()
done = False
screen = None

print("🎮 Watching trained agent... Press 1 or 2 to switch personas. Press ESC or close window to quit.")

# --- Main loop ---
while True:
    # Predict next action
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, info = env.step(action)
    frame = env.render(mode="rgb_array")

    # Create pygame window
    if screen is None:
        screen = pygame.display.set_mode((frame.shape[1], frame.shape[0]))
        pygame.display.set_caption("IsaacLite - Watch Mode")

    # Show frame
    surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
    screen.blit(surf, (0, 0))
    pygame.display.flip()

    # --- Handle input ---
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            exit()
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                pygame.quit()
                exit()
            elif event.key == pygame.K_1:
                current_persona = "survivor"
                print("🔄 Switched to survivor persona")
                model = load_latest_model(current_persona)
                env.close()
                env = IsaacLiteEnv(persona=current_persona)
                obs, _ = env.reset()
            elif event.key == pygame.K_2:
                current_persona = "explorer"
                print("🔄 Switched to explorer persona")
                model = load_latest_model(current_persona)
                env.close()
                env = IsaacLiteEnv(persona=current_persona)
                obs, _ = env.reset()

    # --- Off-screen motivation (if env supports it) ---
    # If your env exposes player position, we can use it like:
    if hasattr(env, "player") and hasattr(env.player, "rect"):
        px, py = env.player.rect.center
        if px < 0 or px > frame.shape[1] or py < 0 or py > frame.shape[0]:
            reward -= 1.0  # small penalty
            print("⚠️ Stay within bounds!")
            time.sleep(0.2)  # brief slowdown as “feedback”

    if done:
        obs, _ = env.reset()

    time.sleep(frame_delay)
