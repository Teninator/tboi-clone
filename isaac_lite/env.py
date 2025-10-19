import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import time
import os
from isaac_lite.game import SimpleGame, ROOM_W, ROOM_H, PLAYER_RADIUS, ENEMY_RADIUS


class IsaacLiteEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}

    def __init__(self, seed=None, persona='survivor', max_steps=200, log_dir="logs"):
        super().__init__()
        self.seed_val = seed if seed is not None else int(time.time())
        self.rng = random.Random(self.seed_val)
        self.game = SimpleGame(rng=self.rng, max_rooms=4)
        self.action_space = spaces.Discrete(9)

        obs_dim = 3 + 9 + 6 + 1
        self.observation_space = spaces.Box(low=-9999, high=9999, shape=(obs_dim,), dtype=np.float32)

        self.persona = persona
        self.max_steps = max_steps
        self.log_dir = log_dir
        os.makedirs(log_dir, exist_ok=True)

        self.episode_id = 0
        self.steps = 0
        self.score = 0.0
        self.powerups = []
        self.active_boosts = {}
        self.death_frame = None
        self.killed_enemy_ids = set()

        # --- Initialize seeds ---
        self.seed(seed)

    # ------------------------------------------------------
    # SEED
    # ------------------------------------------------------
    def seed(self, seed=None):
        """Set seed for reproducibility."""
        self.seed_val = seed if seed is not None else int(time.time())
        self.rng = random.Random(self.seed_val)
        self.np_random, _ = gym.utils.seeding.np_random(self.seed_val)
        random.seed(self.seed_val)
        return [self.seed_val]

    # ------------------------------------------------------
    # RESET
    # ------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        if seed is not None:
            self.seed(seed)
        raw = self.game.reset(seed=self.seed_val)
        self.steps = 0
        self.score = 0.0
        self.powerups = []
        self.active_boosts = {}
        self.death_frame = None
        self.killed_enemy_ids = set()

        # Clear enemy kill flags
        for e in self.game.enemies:
            setattr(e, 'killed_counted', False)

        # Store safe base values for boosts
        self.base_speed = getattr(self.game, "player_speed", 3.0)
        self.base_damage = getattr(self.game, "player_damage", 1.0)

        self.episode_metrics = {
            'time_start': time.time(),
            'time_alive': 0,
            'enemies_killed': 0,
            'rooms_visited': len(raw.get('rooms_visited', [])),
            'damage_taken': 0,
            'shots_fired': 0,
            'deaths': 0
        }
        return self._format_obs(raw), {}

    # ------------------------------------------------------
    # STEP
    # ------------------------------------------------------
    def step(self, action):
        raw, info, done = self.game.step(action)
        self.steps += 1

        # --- Powerup spawn chance ---
        if random.random() < 0.01:
            self._spawn_powerup()

        # --- Apply active boosts ---
        self._update_boosts()

        # --- Reward calculation ---
        reward = self._compute_reward(raw, info, action)
        self.score += reward

        # --- Powerup collection ---
        px, py, _ = raw['player']
        for p in self.powerups[:]:
            if np.hypot(p['x'] - px, p['y'] - py) < PLAYER_RADIUS * 2:
                self._activate_boost(p)
                self.powerups.remove(p)

                # Boost-specific pickup reward
                if p['type'] == 'speed':
                    reward += 2.0
                elif p['type'] == 'damage':
                    reward += 3.0
                self.score += reward

        # --- Track enemy kills safely ---
        info['enemies_killed'] = 0
        for e in self.game.enemies:
            if not e.alive and not getattr(e, 'killed_counted', False):
                setattr(e, 'killed_counted', True)
                self.episode_metrics['enemies_killed'] += 1
                self.score += 0.5
                info['enemies_killed'] += 1

        # --- Reward for using boosts effectively ---
        if "speed" in self.active_boosts:
            reward += 0.01 * (len(raw.get('rooms_visited', [])) - self.episode_metrics.get('rooms_visited', 0))
        if "damage" in self.active_boosts and info['enemies_killed'] > 0:
            reward += 0.5 * info['enemies_killed']

        # --- Death handling ---
        if done and raw['player'][2] <= 0:
            self.episode_metrics['deaths'] += 1
            self.death_frame = time.time()

        obs = self._format_obs(raw)
        self.episode_metrics['time_alive'] = self.steps
        return obs, reward, done, False, info

    # ------------------------------------------------------
    # POWERUPS AND BOOSTS
    # ------------------------------------------------------
    def _spawn_powerup(self):
        """Randomly spawn collectible boosts."""
        types = ["speed", "damage"]
        t = random.choice(types)
        x = random.uniform(50, ROOM_W - 50)
        y = random.uniform(50, ROOM_H - 50)
        self.powerups.append({
            "x": x,
            "y": y,
            "type": t,
            "ttl": random.randint(150, 300)
        })

    def _activate_boost(self, p):
        """Activate temporary boosts (speed or damage)."""
        duration = 200
        boost_type = p["type"]

        # Set safe defaults
        if not hasattr(self.game, "player_speed"):
            self.game.player_speed = getattr(self, "base_speed", 3.0)
        if not hasattr(self.game, "player_damage"):
            self.game.player_damage = getattr(self, "base_damage", 1.0)

        # Refresh duration if already active
        if boost_type in self.active_boosts:
            self.active_boosts[boost_type] = duration
            return

        # Apply the effect
        if boost_type == "speed":
            self.game.player_speed *= 1.5
        elif boost_type == "damage":
            self.game.player_damage *= 1.5

        self.active_boosts[boost_type] = duration

    def _update_boosts(self):
        """Tick down and expire boosts gracefully."""
        expired = []
        for boost, remaining in list(self.active_boosts.items()):
            self.active_boosts[boost] -= 1
            if self.active_boosts[boost] <= 0:
                expired.append(boost)

        for boost in expired:
            if boost == "speed":
                self.game.player_speed = getattr(self, "base_speed", 3.0)
            elif boost == "damage":
                self.game.player_damage = getattr(self, "base_damage", 1.0)
            del self.active_boosts[boost]

        # TTL handling for powerups on map
        for p in self.powerups[:]:
            p["ttl"] -= 1
            if p["ttl"] <= 0:
                self.powerups.remove(p)

    # ------------------------------------------------------
    # REWARD FUNCTION
    # ------------------------------------------------------
    def _compute_reward(self, raw, info, action):
        r = -0.01  # small time penalty per step

        # Base rewards
        r += info.get('enemies_killed', 0) * 0.5
        r -= info.get('damage_taken', 0) * 0.5

        # Persona-specific modifiers
        if self.persona == 'survivor':
            if info.get('damage_taken', 0) == 0:
                r += 0.02
        elif self.persona == 'explorer':
            current_rooms = len(raw.get('rooms_visited', []))
            r += 0.2 * (current_rooms - self.episode_metrics.get('rooms_visited', 0))
            self.episode_metrics['rooms_visited'] = current_rooms

        # Reward slight aggression
        if action in (5, 6, 7, 8):
            r += 0.01

        # Spatial encouragement / margin
        x, y = raw['player'][0], raw['player'][1]
        margin = 0
        if x < margin or x > ROOM_W - margin or y < margin or y > ROOM_H - margin:
            r -= 5.0
        else:
            r += 0.1
            dx = min(x, ROOM_W - x)
            dy = min(y, ROOM_H - y)
            distance_to_edge = min(dx, dy)
            r += 0.01 * distance_to_edge

        # Reward for active boosts
        for boost in self.active_boosts:
            if boost == "speed":
                r += 0.02  # movement efficiency bonus
            elif boost == "damage":
                r += 0.05 * info.get('enemies_killed', 0)  # reward kills while boosted

        return float(r)

    # ------------------------------------------------------
    # OBSERVATION FORMATTING
    # ------------------------------------------------------
    def _format_obs(self, raw):
        px, py, ph = raw['player']
        obs = np.zeros(self.observation_space.shape, dtype=np.float32)
        obs[0] = px
        obs[1] = py
        obs[2] = ph
        return obs

    # ------------------------------------------------------
    # RENDER
    # ------------------------------------------------------
    def render(self, mode='rgb_array'):
        import pygame, numpy as _np
        surf = pygame.Surface((ROOM_W + 150, ROOM_H))
        surf.fill((15, 15, 15))

        # --- Main play area ---
        game_area = pygame.Surface((ROOM_W, ROOM_H))
        game_area.fill((10, 10, 10))

        # Player
        pygame.draw.circle(game_area, (0, 200, 0), (int(self.game.player_x), int(self.game.player_y)), PLAYER_RADIUS)

        # Enemies
        for e in self.game.enemies:
            if e.alive:
                pygame.draw.circle(game_area, (200, 0, 0), (int(e.x), int(e.y)), ENEMY_RADIUS)

        # Bullets — visual boost feedback
        bullet_color = (255, 255, 0)
        bullet_size_mult = 1.0
        if "damage" in self.active_boosts:
            bullet_color = (255, 120, 0)
            bullet_size_mult = 1.4

        for b in self.game.shots:
            pygame.draw.circle(game_area, bullet_color, (int(b.x), int(b.y)), int(b.radius * bullet_size_mult))

        # Power-ups
        for p in self.powerups:
            color = (0, 255, 255) if p['type'] == 'speed' else (255, 0, 255)
            pygame.draw.circle(game_area, color, (int(p['x']), int(p['y'])), 6)
        surf.blit(game_area, (0, 0))

        # --- HUD ---
        font = pygame.font.SysFont("consolas", 18)
        text_lines = [
            f"Score: {int(self.score)}",
            f"Kills: {self.episode_metrics.get('enemies_killed', 0)}",
            f"Boosts: {', '.join(self.active_boosts.keys()) or 'None'}",
            f"Steps: {self.steps}"
        ]
        for i, line in enumerate(text_lines):
            surf.blit(font.render(line, True, (255, 255, 255)), (ROOM_W + 10, 30 + i * 25))

        # --- Death fade ---
        if self.death_frame:
            fade_alpha = self._get_death_fade_alpha()
            if fade_alpha > 0:
                skull_surf = pygame.Surface((ROOM_W, ROOM_H), pygame.SRCALPHA)
                skull_font = pygame.font.SysFont("consolas", 36, bold=True)
                skull_text = skull_font.render("💀 YOU DIED 💀", True, (255, 80, 80))
                skull_text.set_alpha(fade_alpha)
                skull_rect = skull_text.get_rect(center=(ROOM_W // 2, ROOM_H // 2))
                skull_surf.blit(skull_text, skull_rect)

                dim_overlay = pygame.Surface((ROOM_W, ROOM_H), pygame.SRCALPHA)
                dim_overlay.fill((0, 0, 0, 100))
                surf.blit(dim_overlay, (0, 0))
                surf.blit(skull_surf, (0, 0))

        arr = _np.transpose(_np.array(pygame.surfarray.pixels3d(surf)), (1, 0, 2))
        return arr

    # ------------------------------------------------------
    # DEATH FADE
    # ------------------------------------------------------
    def _get_death_fade_alpha(self):
        if not self.death_frame:
            return 0
        elapsed = time.time() - self.death_frame
        fade_duration = 2.5
        if elapsed > fade_duration:
            return 0
        return int(255 * (1 - (elapsed / fade_duration)))

    def close(self):
        pass
