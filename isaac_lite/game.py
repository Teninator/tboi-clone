import random
import math
import numpy as np

ROOM_W, ROOM_H = 640, 480
PLAYER_RADIUS = 12
ENEMY_RADIUS = 10


class Entity:
    def __init__(self, x, y, radius, hp=1):
        self.x = x
        self.y = y
        self.radius = radius
        self.hp = hp
        self.alive = True

    def hit(self, dmg):
        self.hp -= dmg
        if self.hp <= 0:
            self.alive = False


class Bullet:
    def __init__(self, x, y, dx, dy, radius=4, speed=6, lifetime=60):
        self.x = x
        self.y = y
        self.dx = dx
        self.dy = dy
        self.radius = radius
        self.speed = speed
        self.lifetime = lifetime

    def step(self):
        self.x += self.dx * self.speed
        self.y += self.dy * self.speed
        self.lifetime -= 1
        return self.lifetime > 0


class SimpleGame:
    """Game"""

    def __init__(self, rng=None, max_rooms=4):
        self.rng = rng or random.Random()
        self.max_rooms = max_rooms
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng.seed(seed)

        self.player_x = ROOM_W // 2
        self.player_y = ROOM_H // 2
        self.player_hp = 10
        self.player_speed = 3.5
        self.player_damage = 1.0

        self.enemies = []
        self.shots = []
        self.rooms_visited = set([(0, 0)])
        self.frame = 0
        self.spawn_enemy()

        return self._snapshot()

    # ---------------------------------------------------------
    def spawn_enemy(self):
        for _ in range(self.rng.randint(2, 4)):
            ex = self.rng.randint(50, ROOM_W - 50)
            ey = self.rng.randint(50, ROOM_H - 50)
            e = Entity(ex, ey, ENEMY_RADIUS, hp=self.rng.randint(1, 3))
            self.enemies.append(e)

    # ---------------------------------------------------------
    def step(self, action):
        """Performs a game tick based on an integer action."""
        dx = dy = 0

        # Movement
        if action == 0:  # up
            dy = -1
        elif action == 1:  # down
            dy = 1
        elif action == 2:  # left
            dx = -1
        elif action == 3:  # right
            dx = 1

        # Normalize and apply movement
        if dx != 0 or dy != 0:
            mag = math.sqrt(dx * dx + dy * dy)
            dx /= mag
            dy /= mag
            self.player_x += dx * self.player_speed
            self.player_y += dy * self.player_speed

        # Clamp inside room
        self.player_x = max(PLAYER_RADIUS, min(ROOM_W - PLAYER_RADIUS, self.player_x))
        self.player_y = max(PLAYER_RADIUS, min(ROOM_H - PLAYER_RADIUS, self.player_y))

        # Shooting
        if action in (5, 6, 7, 8):
            self.shoot(action)

        # Update bullets
        self.shots = [b for b in self.shots if b.step()]

        info = {"damage_taken": 0, "enemies_killed": 0}

        # Enemy behavior (simple homing)
        for e in self.enemies:
            if not e.alive:
                continue
            ex, ey = e.x, e.y
            dx = self.player_x - ex
            dy = self.player_y - ey
            dist = math.sqrt(dx * dx + dy * dy)
            if dist > 0:
                e.x += (dx / dist) * 1.0
                e.y += (dy / dist) * 1.0

            # Collision with player
            if dist < PLAYER_RADIUS + ENEMY_RADIUS:
                self.player_hp -= 1
                info["damage_taken"] += 1
                if self.player_hp <= 0:
                    break

        # Bullet collision
        for b in self.shots:
            for e in self.enemies:
                if e.alive and math.hypot(b.x - e.x, b.y - e.y) < e.radius + b.radius:
                    e.hit(self.player_damage)
                    if not e.alive:
                        info["enemies_killed"] += 1
                    b.lifetime = 0

        self.shots = [b for b in self.shots if b.lifetime > 0]
        self.frame += 1

        done = self.player_hp <= 0
        return self._snapshot(), info, done

    # ---------------------------------------------------------
    def shoot(self, action):
        # Directions: up=5, down=6, left=7, right=8
        dirs = {
            5: (0, -1),
            6: (0, 1),
            7: (-1, 0),
            8: (1, 0)
        }
        dx, dy = dirs.get(action, (0, 0))
        b = Bullet(self.player_x, self.player_y, dx, dy)
        self.shots.append(b)

    # ---------------------------------------------------------
    def _snapshot(self):
        return {
            "player": (self.player_x, self.player_y, self.player_hp),
            "enemies": [(e.x, e.y, e.alive) for e in self.enemies],
            "rooms_visited": self.rooms_visited,
        }
