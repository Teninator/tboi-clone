import pygame
import json
import time
import os
from isaac_lite.env import IsaacLiteEnv

def play_and_record(output_path="logs/human_sessions/session.json", persona='survivor'):
    pygame.init()
    screen = pygame.display.set_mode((800, 600))
    pygame.display.set_caption("Isaac Lite - Human Play Mode")
    clock = pygame.time.Clock()

    env = IsaacLiteEnv(persona=persona)
    obs, info = env.reset()
    done = False
    data = []
    font = pygame.font.SysFont("consolas", 24)

    key_to_action = {
        pygame.K_UP: 0,
        pygame.K_DOWN: 1,
        pygame.K_LEFT: 2,
        pygame.K_RIGHT: 3,
        pygame.K_SPACE: 4
    }

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    print("🎮 Controls: Arrow keys to move, SPACE to shoot, ESC to quit.")

    # Game loop
    running = True
    while running:
        action = 8  # default idle
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE:
                running = False

        keys = pygame.key.get_pressed()
        for k, a in key_to_action.items():
            if keys[k]:
                action = a

        obs, reward, done, _, info = env.step(action)
        frame = env.render()

        # Draw on screen
        surf = pygame.surfarray.make_surface(frame.swapaxes(0, 1))
        screen.blit(surf, (0, 0))

        # Show info
        text_surface = font.render(f"Reward: {reward:.2f} | Score: {env.score:.1f}", True, (255, 255, 255))
        screen.blit(text_surface, (20, 20))

        pygame.display.flip()
        clock.tick(30)

        # Record data
        data.append({
            "obs": obs.tolist(),
            "action": int(action),
            "reward": float(reward)
        })

        # Handle win/loss screens with pause
        if env.win_frame or env.death_frame:
            pygame.display.flip()
            end_message = "🎉 YOU WON! 🎉" if env.win_frame else "💀 YOU DIED 💀"
            color = (255, 255, 0) if env.win_frame else (255, 80, 80)
            text = font.render(end_message, True, color)
            rect = text.get_rect(center=(400, 300))
            screen.blit(text, rect)
            pygame.display.flip()
            time.sleep(3)
            obs, info = env.reset()

    # Cleanup
    env.close()
    pygame.quit()

    # Save data
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"Human session saved: {output_path}")

if __name__ == "__main__":
    timestamp = int(time.time())
    output_file = f"logs/human_sessions/session_{timestamp}.json"
    play_and_record(output_path=output_file)
