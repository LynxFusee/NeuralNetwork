"""
Car Dodge — a simple survival game, built to be AI-controllable.

The car moves left/right along the bottom of the screen while obstacles
fall from the top. Survive as long as possible.

The file is split into two layers on purpose:

1. CarDodgeEnv — pure game logic (state, physics, collisions, reward).
   This is what your AI will actually talk to. It doesn't need a window
   or a human anywhere near it.
2. A pygame front-end (`run()`) so a human can play, or so you can watch
   an AI play by passing it an `agent` function.

HOW AN AI PLUGS IN
-------------------
    env = CarDodgeEnv()
    state = env.reset()
    while True:
        action = your_ai.choose_action(state)   # 0 = stay, 1 = left, 2 = right
        state, reward, done, info = env.step(action)
        if done:
            break

`state` is a flat list of floats: the car's position plus the next few
obstacles' positions/speeds, all normalized to roughly 0-1. That's a
ready-made input vector for a Q-table, a small neural net, a genetic
algorithm — whatever you build next. `reward` is +0.1 for every frame
survived and -10 on crash, which is enough to get reinforcement learning
off the ground (maximize total reward = survive longer).

RUNNING IT (NixOS)
-------------------
    nix-shell -p python3 python3Packages.pygame --run "python car_dodge.py"

CONTROLS
--------
Arrow keys move the car (safest bet given your AZERTY layout — the
letter-key aliases below assume the physical key produces 'q'/'d',
which is where your fingers rest on AZERTY; swap them in the code if
that's not the case for you).
"""

import sys
import random
import pygame
import ai

# ---------------------------------------------------------------- #
# Config — tweak freely
# ---------------------------------------------------------------- #
SCREEN_W, SCREEN_H = 480, 640
CAR_W, CAR_H = 50, 80
CAR_Y = SCREEN_H - CAR_H - 20
CAR_SPEED = 8

OBSTACLE_W, OBSTACLE_H = 50, 50
OBSTACLE_START_SPEED = 5
OBSTACLE_SPEED_RAMP = 0.09       # obstacle speed grows with survival time
SPAWN_EVERY_MS_START = 900
SPAWN_EVERY_MS_MIN = 350
SPAWN_RAMP = 20                  # spawn interval shrinks with survival time

N_TRACKED_OBSTACLES = 3          # how many upcoming obstacles feed into the state vector
FPS = 60

ACTION_STAY, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2


class CarDodgeEnv:
    """Pure game logic. No window required to call reset()/step()."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.car_x = SCREEN_W / 2 - CAR_W / 2
        self.obstacles = []          # each entry: [x, y, speed]
        self.time_alive = 0.0
        self.spawn_timer = 0.0
        self.spawn_interval = SPAWN_EVERY_MS_START
        self.obstacle_speed = OBSTACLE_START_SPEED
        self.done = False
        return self._get_state()

    def step(self, action, dt_ms=1000 / FPS):
        if self.done:
            raise RuntimeError("Episode finished — call reset() before stepping again.")

        # 1. move the car
        if action == ACTION_LEFT:
            self.car_x -= CAR_SPEED
        elif action == ACTION_RIGHT:
            self.car_x += CAR_SPEED
        self.car_x = max(0, min(SCREEN_W - CAR_W, self.car_x))

        # 2. ramp difficulty over time
        self.time_alive += dt_ms / 1000
        self.obstacle_speed = OBSTACLE_START_SPEED + self.time_alive * OBSTACLE_SPEED_RAMP
        self.spawn_interval = max(
            SPAWN_EVERY_MS_MIN,
            SPAWN_EVERY_MS_START - self.time_alive * SPAWN_RAMP,
        )

        # 3. spawn new obstacles
        self.spawn_timer += dt_ms
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            x = random.randint(0, SCREEN_W - OBSTACLE_W)
            self.obstacles.append([x, -OBSTACLE_H, self.obstacle_speed])

        # 4. move obstacles, drop the ones that left the screen
        for ob in self.obstacles:
            ob[1] += ob[2]
        self.obstacles = [ob for ob in self.obstacles if ob[1] < SCREEN_H]

        # 5. collision check
        car_rect = pygame.Rect(self.car_x, CAR_Y, CAR_W, CAR_H)
        collided = any(
            car_rect.colliderect(pygame.Rect(ob[0], ob[1], OBSTACLE_W, OBSTACLE_H))
            for ob in self.obstacles
        )

        reward = 0.1
        if collided:
            self.done = True
            reward = -10.0

        return self._get_state(), reward, self.done, {"time_alive": self.time_alive}

    def _get_state(self):
        """car_x (normalized) + the N nearest obstacles' (x, y, speed), normalized."""
        state = [self.car_x / SCREEN_W]
        upcoming = sorted(self.obstacles, key=lambda ob: ob[1])[:N_TRACKED_OBSTACLES]
        for ob in upcoming:
            state += [ob[0] / SCREEN_W, ob[1] / SCREEN_H, ob[2] / 20]
        while len(state) < 1 + N_TRACKED_OBSTACLES * 3:
            state += [0.0, -1.0, 0.0]     # padding when fewer obstacles exist yet
        return state


# ---------------------------------------------------------------- #
# Pygame front-end
# ---------------------------------------------------------------- #
def run(agent=None):
    """
    agent=None              -> keyboard control (you play)
    agent=callable(state)   -> agent(state) must return 0/1/2; used to watch an AI play
    """
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Car Dodge")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)

    env = CarDodgeEnv()
    state = env.reset()

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        if agent is not None:
            action = agent(state)
        else:
            keys = pygame.key.get_pressed()
            if keys[pygame.K_LEFT] or keys[pygame.K_a]:
                action = ACTION_LEFT
            elif keys[pygame.K_RIGHT] or keys[pygame.K_d]:
                action = ACTION_RIGHT
            else:
                action = ACTION_STAY

        state, reward, done, info = env.step(action)

        screen.fill((25, 25, 35))
        pygame.draw.rect(screen, (60, 200, 90), (env.car_x, CAR_Y, CAR_W, CAR_H))
        for ob in env.obstacles:
            pygame.draw.rect(screen, (220, 70, 70), (ob[0], ob[1], OBSTACLE_W, OBSTACLE_H))

        timer_surf = font.render(f"Time: {info['time_alive']:.1f}s", True, (230, 230, 230))
        screen.blit(timer_surf, (10, 10))

        if done:
            msg = font.render("Game over — press R to restart", True, (255, 255, 255))
            screen.blit(msg, (SCREEN_W / 2 - msg.get_width() / 2, SCREEN_H / 2))
            pygame.display.flip()
            waiting = True
            while waiting:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        sys.exit()
                    if event.type == pygame.KEYDOWN and event.key == pygame.K_r:
                        state = env.reset()
                        waiting = False
            continue

        pygame.display.flip()
        clock.tick(FPS)


def agent(state):
    """Placeholder AI — picks a random action. Swap this out for your own logic."""
    result = net.logic(state)
    if result <= -0.66 :
        return ACTION_LEFT
    if result >= 0.66 :
        return ACTION_RIGHT
    return ACTION_STAY


if __name__ == "__main__":
    # Switch to run(agent=random_agent_demo) to watch a (very dumb) AI play instead.
    net = ai.Network(3,10,None, None, 9, 1)
    net.createNetwork()
    run()
    #run(agent=random_agent_demo)