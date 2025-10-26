import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
import json
import os
import pygame
from isaac_lite.game import SimpleGame, ROOM_W, ROOM_H, PLAYER_RADIUS, ENEMY_RADIUS


class IsaacLiteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, seed=None, persona='survivor', max_steps=200, log_dir="logs"):
        super().__init__()
        self.seed_val = seed if seed is not None else int(time.time())
        self.rng = random.Random(self.seed_val)

        self.game = SimpleGame(rng=self.rng, max_rooms=4)
        self.action_space = spaces.Discrete(9)

        obs_dim = 20
        self.observation_space = spaces.Box(low=-9999, high=9999, shape=(obs_dim,), dtype=np.float32)

        self.persona = persona
        self.max_steps = max_steps
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        # Game state
        self.steps = 0
        self.score = 0.0
        self.powerups = []               
        self.active_boosts = {}         
        self.death_frame = None
        self.win_frame = None
        self.confetti_particles = []
        self.killed_enemy_ids = set()

        # Player stat baselines (defensive if missing in game)
        self.base_speed = getattr(self.game, "player_speed", 3.0)
        self.base_damage = getattr(self.game, "player_damage", 1.0)

        self.seed(seed)

    # SEED
    def seed(self, seed=None):
        self.seed_val = seed if seed is not None else int(time.time())
        self.rng = random.Random(self.seed_val)
        self.np_random, _ = gym.utils.seeding.np_random(self.seed_val)
        random.seed(self.seed_val)
        return [self.seed_val]

    # RESET
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)

        raw = self.game.reset(seed=self.seed_val)

        self.steps = 0
        self.score = 0.0
        self.powerups.clear()
        self.active_boosts.clear()
        self.death_frame = None
        self.win_frame = None
        self.confetti_particles.clear()
        self.killed_enemy_ids.clear()

        self.episode_metrics = 
        {
            'time_start': time.time(),
            'time_alive': 0,
            'enemies_killed': 0,
            'rooms_visited': len(raw.get('rooms_visited', [])),
            'damage_taken': 0,
            'shots_fired': 0,
            'deaths': 0
        }

        # For movement reward
        self.last_pos = (raw['player'][0], raw['player'][1])

        return self._format_obs(raw), {}

    # STEP
    def step(self, action: int):
        raw, info, done = self.game.step(action)
        self.steps += 1

        # Random powerup spawns
        if random.random() < 0.01:
            self._spawn_powerup()

        # Decay / expire active boosts
        self._update_boosts()

        # Reward shaping
        reward = self._compute_reward(raw, info, action)
        self.score += reward

        # Handle powerup pickup
        px, py, _ = raw['player']
        for p in self.powerups[:]:
            if np.hypot(p['x'] - px, p['y'] - py) < PLAYER_RADIUS * 2:
                self._activate_boost(p)
                self.powerups.remove(p)
                reward += 2.0
                self.score += 2.0

        # Count kills once
        info['enemies_killed'] = 0
        for e in self.game.enemies:
            if not e.alive and e not in self.killed_enemy_ids:
                self.killed_enemy_ids.add(e)
                self.episode_metrics['enemies_killed'] += 1
                self.score += 0.5
                info['enemies_killed'] += 1

        # Win condition (all enemies dead)
        if all(not e.alive for e in self.game.enemies):
            if not self.win_frame:
                self.win_frame = time.time()
                self._spawn_confetti()

        # Death condition
        if done and raw['player'][2] <= 0:
            self.episode_metrics['deaths'] += 1
            self.death_frame = time.time()

        obs = self._format_obs(raw)
        self.episode_metrics['time_alive'] = self.steps
        # truncated is False here; SimpleGame decides 'done'
        return obs, float(reward), bool(done), False, info

    # POWERUPS & BOOSTS
    def _spawn_powerup(self):
        t = random.choice(["speed", "damage"])
        self.powerups.append({
            "x": random.uniform(50, ROOM_W - 50),
            "y": random.uniform(50, ROOM_H - 50),
            "type": t,
            "ttl": random.randint(150, 300)
        })

    def _activate_boost(self, p):
        duration = 200
        boost_type = p["type"]

        # Refresh if already active
        if boost_type in self.active_boosts:
            self.active_boosts[boost_type] = duration
            return

        # Apply stat change
        if boost_type == "speed":
            self.game.player_speed = self.base_speed * 1.5
        elif boost_type == "damage":
            self.game.player_damage = self.base_damage * 1.5

        self.active_boosts[boost_type] = duration

    def _update_boosts(self):
        expired = []
        for b in list(self.active_boosts.keys()):
            self.active_boosts[b] -= 1
            if self.active_boosts[b] <= 0:
                expired.append(b)

        for b in expired:
            if b == "speed":
                self.game.player_speed = self.base_speed
            elif b == "damage":
                self.game.player_damage = self.base_damage
            del self.active_boosts[b]

        # Powerup TTL
        for p in self.powerups[:]:
            p["ttl"] -= 1
            if p["ttl"] <= 0:
                self.powerups.remove(p)

    # REWARD FUNCTION
    def _compute_reward(self, raw, info, action):
        # Base shaping
        r = -0.01  # discourage idling

        # Combat terms
        r += info.get('enemies_killed', 0) * 0.6
        r -= info.get('damage_taken', 0) * 0.4

        # Persona-specific
        if self.persona == 'survivor':
            if info.get('damage_taken', 0) == 0:
                r += 0.02
        elif self.persona == 'explorer':
            current_rooms = len(raw.get('rooms_visited', []))
            prev_rooms = self.episode_metrics.get('rooms_visited', 0)
            if current_rooms > prev_rooms:
                r += 0.4
                self.episode_metrics['rooms_visited'] = current_rooms

        # Movement reward
        px, py = raw['player'][0], raw['player'][1]
        dx = px - self.last_pos[0]
        dy = py - self.last_pos[1]
        dist_moved = np.sqrt(dx * dx + dy * dy)
        self.last_pos = (px, py)

        if dist_moved > 1.0:
            r += 0.05 * dist_moved
        else:
            r -= 0.01

        # Encourage directional movement and shooting a bit
        if action in (0, 1, 2, 3):   # movement actions
            r += 0.03
        elif action in (5, 6, 7, 8): # shooting actions
            r += 0.01

        # Boundary logic
        if px < 0 or px > ROOM_W or py < 0 or py > ROOM_H:
            r -= 2.0  # touching / leaving bounds
        else:
            r += 0.02

        # Active boost encouragement
        for b in self.active_boosts:
            if b == "speed":
                r += 0.02
            elif b == "damage":
                r += 0.05 * info.get('enemies_killed', 0)

        # Survival trickle
        r += 0.03

        # Cleans up rewards
        return float(np.clip(r, -2.0, 2.0))

    # RENDER
    def render(self, mode='rgb_array'):
        # Creates an off-screen surface and draw the whole HUD+world
        surf = pygame.Surface((ROOM_W + 150, ROOM_H))
        surf.fill((10, 10, 10))

        # World
        game_area = pygame.Surface((ROOM_W, ROOM_H))
        game_area.fill((15, 15, 20))

        # Player
        pygame.draw.circle(
            game_area,
            (0, 200, 0),
            (int(self.game.player_x), int(self.game.player_y)),
            PLAYER_RADIUS
        )

        # Enemies
        for e in self.game.enemies:
            if e.alive:
                pygame.draw.circle(game_area, (200, 0, 0), (int(e.x), int(e.y)), ENEMY_RADIUS)

        # Bullets (turns redish orange when damage-boosted)
        bullet_color = (255, 255, 0) if "damage" not in self.active_boosts else (255, 120, 0)
        for b in self.game.shots:
            pygame.draw.circle(game_area, bullet_color, (int(b.x), int(b.y)), 4)

        # Powerups
        for p in self.powerups:
            color = (0, 255, 255) if p['type'] == 'speed' else (255, 0, 255)
            pygame.draw.circle(game_area, color, (int(p['x']), int(p['y'])), 6)

        surf.blit(game_area, (0, 0))

        # HUD
        font = pygame.font.SysFont("consolas", 18)

        lines = [
            f"Score: {int(self.score)}",
            f"Kills: {self.episode_metrics.get('enemies_killed', 0)}",
            f"Boosts: {', '.join(self.active_boosts.keys()) or 'None'}",
            f"Steps: {self.steps}"
        ]
        for i, l in enumerate(lines):
            txt = font.render(l, True, (255, 255, 255))
            surf.blit(txt, (ROOM_W + 10, 30 + i * 25))

        # Death fade
        if self.death_frame:
            fade_alpha = self._get_death_fade_alpha()
            skull_font = pygame.font.SysFont("consolas", 36, bold=True)
            skull_text = skull_font.render("YOU DIED", True, (255, 80, 80))
            skull_text.set_alpha(fade_alpha)
            surf.blit(skull_text, skull_text.get_rect(center=(ROOM_W // 2, ROOM_H // 2)))

        # Win
        if self.win_frame:
            self._update_confetti()
            self._draw_confetti(surf)
            win_font = pygame.font.SysFont("consolas", 36, bold=True)
            win_text = win_font.render("YOU WON!", True, (255, 255, 100))
            surf.blit(win_text, win_text.get_rect(center=(ROOM_W // 2, ROOM_H // 2)))

        # Return RGB array (H, W, 3)
        arr = np.transpose(np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2))
        return arr

    # CONFETTI EFFECTS
    def _spawn_confetti(self):
        self.confetti_particles = []
        for _ in range(80):
            self.confetti_particles.append({
                "x": random.randint(0, ROOM_W),
                "y": random.randint(0, ROOM_H),
                "vx": random.uniform(-2, 2),
                "vy": random.uniform(-5, -1),
                "color": (
                    random.randint(100, 255),
                    random.randint(100, 255),
                    random.randint(100, 255)
                ),
                "life": random.randint(30, 60)
            })

    def _update_confetti(self):
        for c in self.confetti_particles[:]:
            c["x"] += c["vx"]
            c["y"] += c["vy"]
            c["vy"] += 0.2
            c["life"] -= 1
            if c["life"] <= 0:
                self.confetti_particles.remove(c)

    def _draw_confetti(self, surface):
        for c in self.confetti_particles:
            pygame.draw.circle(surface, c["color"], (int(c["x"]), int(c["y"])), 3)

    # OBSERVATION VECTOR
    def _format_obs(self, raw):
        px, py, ph = raw.get('player', (0, 0, 0))
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)

        # player
        obs[0:3] = [px, py, ph]

        # enemies (up to 3)
        enemies = raw.get("enemies", [])
        for i, e in enumerate(enemies[:3]):
            if isinstance(e, tuple):
                ex, ey, alive = e if len(e) == 3 else (e[0], e[1], 1.0)
            else:
                ex = getattr(e, "x", 0.0)
                ey = getattr(e, "y", 0.0)
                alive = 1.0 if getattr(e, "alive", True) else 0.0
            obs[3 + i * 3: 6 + i * 3] = [ex, ey, alive]

        # powerups (up to 2) -> indices 12..17
        for i, p in enumerate(self.powerups[:2]):
            obs[12 + i * 3: 15 + i * 3] = [p["x"], p["y"], 1.0]

        # boost flags -> indices 18,19
        obs[18] = 1.0 if "damage" in self.active_boosts else 0.0
        obs[19] = 1.0 if "speed" in self.active_boosts else 0.0

        return obs

    # UTILITIES
    def _get_death_fade_alpha(self):
        if not self.death_frame:
            return 0
        elapsed = time.time() - self.death_frame
        fade_duration = 2.5
        return int(255 * max(0.0, 1.0 - elapsed / fade_duration))

    def save_episode_metrics(self, filename="episode_metrics.json"):
        path = os.path.join(self.log_dir, filename)

        with open(path, "w") as f:
            json.dump(self.episode_metrics, f, indent=2)

        print(f"Saved metrics to {path} :)")

    
    def close(self):
        pass
