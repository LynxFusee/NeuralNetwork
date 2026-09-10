import sys
import random
import pygame
import ai
import copy
import json
import os

# ==========================================
#         PARAMÈTRES GLOBAUX
# ==========================================

NUM_ENVS = 5                   
CARS_PER_ENV = 60              
MUTATION_RATE = 0.05           
FPS = 60                       

GAME_W, GAME_H = 250, 450      
WINDOW_W = GAME_W * NUM_ENVS   
# On ajoute 60 pixels en bas pour l'affichage des statistiques
WINDOW_H = GAME_H + 60         

CAR_W, CAR_H = 30, 45          
CAR_Y = GAME_H - CAR_H - 10
CAR_SPEED = 5
ACTION_STAY, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2

OBSTACLE_W, OBSTACLE_H = 35, 35
OBSTACLE_START_SPEED = 4
OBSTACLE_SPEED_RAMP = 0.08
SPAWN_EVERY_MS_START = 800
SPAWN_EVERY_MS_MIN = 300
SPAWN_RAMP = 20
N_TRACKED_OBSTACLES = 3        

# ==========================================


class CarDodgeEnv:
    def __init__(self, num_cars=CARS_PER_ENV):
        self.num_cars = num_cars
        self.reset()

    def reset(self):
        self.cars = [{"x": GAME_W / 2 - CAR_W / 2, "alive": True, "time_alive": 0.0} for _ in range(self.num_cars)]
        self.obstacles = []
        self.global_time = 0.0
        self.spawn_timer = 0.0
        self.spawn_interval = SPAWN_EVERY_MS_START
        self.obstacle_speed = OBSTACLE_START_SPEED
        self.done = False
        return self.get_states()

    def get_states(self):
        states = []
        for car in self.cars:
            if not car["alive"]:
                states.append(None)
                continue
            
            state = [car["x"] / GAME_W]
            upcoming = sorted(self.obstacles, key=lambda ob: ob[1])[:N_TRACKED_OBSTACLES]
            for ob in upcoming:
                state += [ob[0] / GAME_W, ob[1] / GAME_H, ob[2] / 20]
            while len(state) < 1 + N_TRACKED_OBSTACLES * 3:
                state += [0.0, -1.0, 0.0]
            states.append(state)
        return states

    def step(self, actions, dt_ms=1000 / FPS):
        if self.done: return self.get_states(), True

        self.global_time += dt_ms / 1000
        self.obstacle_speed = OBSTACLE_START_SPEED + self.global_time * OBSTACLE_SPEED_RAMP
        self.spawn_interval = max(SPAWN_EVERY_MS_MIN, SPAWN_EVERY_MS_START - self.global_time * SPAWN_RAMP)

        self.spawn_timer += dt_ms
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            x = random.randint(0, GAME_W - OBSTACLE_W)
            self.obstacles.append([x, -OBSTACLE_H, self.obstacle_speed])

        for ob in self.obstacles:
            ob[1] += ob[2]
        self.obstacles = [ob for ob in self.obstacles if ob[1] < GAME_H]

        all_dead = True
        for i, car in enumerate(self.cars):
            if not car["alive"]:
                continue
            
            act = actions[i]
            if act == ACTION_LEFT:
                car["x"] -= CAR_SPEED
            elif act == ACTION_RIGHT:
                car["x"] += CAR_SPEED
            car["x"] = max(0, min(GAME_W - CAR_W, car["x"]))

            car["time_alive"] += dt_ms / 1000

            car_rect = pygame.Rect(car["x"], CAR_Y, CAR_W, CAR_H)
            collided = any(car_rect.colliderect(pygame.Rect(ob[0], ob[1], OBSTACLE_W, OBSTACLE_H)) for ob in self.obstacles)
            
            if collided:
                car["alive"] = False
            else:
                all_dead = False

        self.done = all_dead
        return self.get_states(), self.done


def train_ai():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(f"Car Dodge - Asynchrone x10 Vitesse")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    
    envs = [CarDodgeEnv(num_cars=CARS_PER_ENV) for _ in range(NUM_ENVS)]
    best_nets = []
    all_networks = []
    generations = [1] * NUM_ENVS
    
    # Nouvelles listes pour traquer les statistiques
    history_bests = [[] for _ in range(NUM_ENVS)]
    all_time_bests = [0.0] * NUM_ENVS

    for env_id in range(NUM_ENVS):
        fichier_env = f"ia_env_{env_id}.json"
        
        if os.path.exists(fichier_env):
            with open(fichier_env, "r") as f:
                data = json.load(f)
            champion = ai.Network(
                layers_count=data["layers_count"], per_layer=data["per_layer"],
                wheights=data["wheights"], bias=data["bias"],
                entry_size=data["entry_size"], exit_size=data["exit_size"]
            )
            champion.createNetwork()
            best_nets.append(champion)
        else:
            net = ai.Network(layers_count=1, per_layer=6, wheights=None, bias=None, entry_size=10, exit_size=1)
            net.createNetwork()
            best_nets.append(net)
            
        family = []
        for i in range(CARS_PER_ENV):
            new_net = copy.deepcopy(best_nets[env_id])
            if i > 0:
                new_net.mutNetwork(MUTATION_RATE)
            family.append(new_net)
        all_networks.append(family)
        
    all_states = [env.reset() for env in envs]
    frame_count = 0
    
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
        
        for env_id in range(NUM_ENVS):
            if envs[env_id].done:
                best_score = -1
                best_index = 0
                for i, car in enumerate(envs[env_id].cars):
                    if car["time_alive"] > best_score:
                        best_score = car["time_alive"]
                        best_index = i
                
                # Sauvegarde du score pour les statistiques
                history_bests[env_id].append(best_score)
                if best_score > all_time_bests[env_id]:
                    all_time_bests[env_id] = best_score
                
                best_nets[env_id] = copy.deepcopy(all_networks[env_id][best_index])
                with open(f"ia_env_{env_id}.json", "w") as f:
                    json.dump(best_nets[env_id].export_json(), f, indent=4)
                
                generations[env_id] += 1
                family = []
                for i in range(CARS_PER_ENV):
                    new_net = copy.deepcopy(best_nets[env_id])
                    if i > 0:
                        new_net.mutNetwork(MUTATION_RATE)
                    family.append(new_net)
                all_networks[env_id] = family
                
                all_states[env_id] = envs[env_id].reset()
                
            else:
                actions = []
                for i in range(CARS_PER_ENV):
                    if envs[env_id].cars[i]["alive"]:
                        state = all_states[env_id][i]
                        result = all_networks[env_id][i].logic(state)[0]
                        if result <= -0.33:
                            actions.append(ACTION_LEFT)
                        elif result >= 0.33:
                            actions.append(ACTION_RIGHT)
                        else:
                            actions.append(ACTION_STAY)
                    else:
                        actions.append(ACTION_STAY)
                        
                all_states[env_id], _ = envs[env_id].step(actions)
        
        frame_count += 1
        if frame_count % 10 == 0:
            screen.fill((20, 20, 20))
            
            for env_id in range(NUM_ENVS):
                offset_x = env_id * GAME_W 
                pygame.draw.line(screen, (100, 100, 100), (offset_x, 0), (offset_x, WINDOW_H), 2)
                
                for ob in envs[env_id].obstacles:
                    pygame.draw.rect(screen, (220, 70, 70), (offset_x + ob[0], ob[1], OBSTACLE_W, OBSTACLE_H))
                
                alives = 0
                for i, car in enumerate(envs[env_id].cars):
                    if car["alive"] and i > 0:
                        alives += 1
                        pygame.draw.rect(screen, (60, 200, 90), (offset_x + car["x"], CAR_Y, CAR_W, CAR_H))
                
                if envs[env_id].cars[0]["alive"]:
                    alives += 1
                    pygame.draw.rect(screen, (60, 200, 255), (offset_x + envs[env_id].cars[0]["x"], CAR_Y, CAR_W, CAR_H))
                
                txt = font.render(f"F{env_id+1} G{generations[env_id]} : {alives} en vie", True, (255, 255, 255))
                screen.blit(txt, (offset_x + 5, 10))

                # --- NOUVEAU : PANNEAU DE STATISTIQUES EN BAS ---
                pygame.draw.rect(screen, (35, 35, 45), (offset_x, GAME_H, GAME_W, 60))
                
                # Calcul de la moyenne des 10 derniers champions
                if len(history_bests[env_id]) > 0:
                    recent = history_bests[env_id][-10:]
                    avg_score = sum(recent) / len(recent)
                else:
                    avg_score = 0.0
                    
                record = all_time_bests[env_id]

                txt_avg = font.render(f"Moyenne (10g) : {avg_score:.1f}s", True, (150, 200, 255))
                txt_rec = font.render(f"Record absolu : {record:.1f}s", True, (255, 215, 0))
                
                screen.blit(txt_avg, (offset_x + 5, GAME_H + 10))
                screen.blit(txt_rec, (offset_x + 5, GAME_H + 35))
            
            pygame.display.flip()
        
        clock.tick(0) 

if __name__ == "__main__":
    train_ai()