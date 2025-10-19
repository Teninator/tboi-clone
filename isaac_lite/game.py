# isaac_lite/game.py
import pygame
import random
import math
import numpy as np

SCREEN_W, SCREEN_H = 160, 160
ROOM_W, ROOM_H = SCREEN_W, SCREEN_H
PLAYER_RADIUS = 6
ENEMY_RADIUS = 6
BULLET_SPEED = 6
PLAYER_SPEED = 3
ENEMY_SPEED = 1.2

class Bullet:
    def __init__(self, x, y, vx, vy, owner):
        self.x = x; self.y = y; self.vx = vx; self.vy = vy; self.owner = owner
        self.radius = 2
        self.alive = True

    def step(self):
        self.x += self.vx
        self.y += self.vy
        if self.x < 0 or self.x > ROOM_W or self.y < 0 or self.y > ROOM_H:
            self.alive = False

class Enemy:
    def __init__(self, x, y, hp=3):
        self.x = x; self.y = y; self.hp = hp; self.radius = ENEMY_RADIUS; self.alive = True

    def step_toward(self, px, py):
        dx = px - self.x; dy = py - self.y
        dist = math.hypot(dx, dy) + 1e-6
        self.x += (dx / dist) * ENEMY_SPEED
        self.y += (dy / dist) * ENEMY_SPEED

class SimpleGame:
    def __init__(self, rng=None, max_rooms=4):
        self.rng = rng or random.Random()
        self.max_rooms = max_rooms
        self.reset()

    def reset(self, seed=None):
        if seed is not None:
            self.rng = random.Random(seed)
        self.player_x = ROOM_W/2
        self.player_y = ROOM_H/2
        self.player_hp = 10
        self.shots = []
        self.enemies = []
        self.spawn_enemies(3)
        self.steps = 0
        self.rooms_visited = {0: True}
        self.current_room = 0
        self.total_rooms = min(self.max_rooms, 6)
        return self._get_raw_obs()

    def spawn_enemies(self, n):
        for _ in range(n):
            x = self.rng.uniform(10, ROOM_W-10)
            y = self.rng.uniform(10, ROOM_H-10)
            self.enemies.append(Enemy(x, y, hp=3))

    def _get_raw_obs(self):
        enemies = [(e.x, e.y, e.hp) for e in self.enemies if e.alive]
        bullets = [(b.x, b.y, b.owner) for b in self.shots if b.alive]
        return {
            'player': (self.player_x, self.player_y, self.player_hp),
            'enemies': enemies,
            'bullets': bullets,
            'rooms_visited': list(self.rooms_visited.keys()),
            'steps': self.steps,
            'room': self.current_room
        }

    def step(self, action):
        self.steps += 1
        dir_map = {
            0: (0,0, False),
            1: (-1, 0, False),
            2: (1, 0, False),
            3: (0, -1, False),
            4: (0, 1, False),
            5: (-1, -1, True),
            6: (1, -1, True),
            7: (-1, 1, True),
            8: (1, 1, True)
        }
        dx, dy, shoot = dir_map.get(int(action), (0,0,False))
        hyp = math.hypot(dx, dy)
        if hyp > 0:
            self.player_x = max(0, min(ROOM_W, self.player_x + (dx/hyp)*PLAYER_SPEED))
            self.player_y = max(0, min(ROOM_H, self.player_y + (dy/hyp)*PLAYER_SPEED))
        if shoot:
            sx, sy = (dx, dy) if hyp>0 else (0, -1)
            vx = sx * BULLET_SPEED if sx!=0 else 0
            vy = sy * BULLET_SPEED if sy!=0 else -BULLET_SPEED
            self.shots.append(Bullet(self.player_x, self.player_y, vx, vy, 'player'))
        for e in self.enemies:
            if e.alive:
                e.step_toward(self.player_x, self.player_y)
        for b in list(self.shots):
            b.step()
            if not b.alive:
                continue
            if b.owner == 'player':
                for e in self.enemies:
                    if not e.alive: continue
                    if (b.x - e.x)**2 + (b.y - e.y)**2 < (b.radius + e.radius)**2:
                        e.hp -= 1
                        b.alive = False
                        if e.hp <= 0:
                            e.alive = False
        damage = 0
        for e in self.enemies:
            if not e.alive: continue
            if (e.x - self.player_x)**2 + (e.y - self.player_y)**2 < (e.radius + PLAYER_RADIUS)**2:
                self.player_hp -= 1
                damage += 1
                dx = self.player_x - e.x; dy = self.player_y - e.y
                d = math.hypot(dx, dy) + 1e-6
                self.player_x += (dx/d) * 4
                self.player_y += (dy/d) * 4
        self.shots = [b for b in self.shots if b.alive]
        done = False
        reward_info = {
            'damage_taken': damage,
            'enemies_alive': sum(1 for e in self.enemies if e.alive),
            'enemies_killed': sum(1 for e in self.enemies if not e.alive)
        }
        if self.player_hp <= 0:
            done = True
        if reward_info['enemies_alive'] == 0 and self.current_room < self.total_rooms-1:
            self.current_room += 1
            self.rooms_visited[self.current_room] = True
            self.spawn_enemies(2 + self.current_room)
        if self.steps >= 200:
            done = True
        return self._get_raw_obs(), reward_info, done
