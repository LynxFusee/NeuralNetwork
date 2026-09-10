import sys
import random
import pygame
import ai
import copy
import json
import os
import multiprocessing as mp

# ==========================================
#         PARAMÈTRES GLOBAUX
# ==========================================
NUM_ENVS = 5                   # Change ça pour le nombre de cœurs de ton CPU (ex: 8)
CARS_PER_ENV = 60              
MUTATION_RATE = 0.05           
FPS = 60                       

GAME_W, GAME_H = 250, 450      
WINDOW_W = GAME_W * NUM_ENVS   
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


def worker_training_loop(env_id, queue_vers_affichage):
    """Tourne sur un cœur CPU dédié, sans limite de vitesse."""
    env = CarDodgeEnv(num_cars=CARS_PER_ENV)
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
    else:
        champion = ai.Network(layers_count=1, per_layer=6, wheights=None, bias=None, entry_size=10, exit_size=1)
        champion.createNetwork()
        
    generation = 1
    step_count = 0
    all_time_best = 0.0
    history_bests = []
    
    while True:
        family = []
        for i in range(CARS_PER_ENV):
            new_net = copy.deepcopy(champion)
            if i > 0: new_net.mutNetwork(MUTATION_RATE)
            family.append(new_net)
            
        states = env.reset()
        
        while not env.done:
            actions = []
            for i in range(CARS_PER_ENV):
                if env.cars[i]["alive"]:
                    state = states[i]
                    result = family[i].logic(state)[0]
                    if result <= -0.33: actions.append(ACTION_LEFT)
                    elif result >= 0.33: actions.append(ACTION_RIGHT)
                    else: actions.append(ACTION_STAY)
                else:
                    actions.append(ACTION_STAY)
                    
            states, _ = env.step(actions)
            step_count += 1
            
            # Envoyer les données visuelles 1 fois sur 20 pour ne pas exploser la RAM
            if step_count % 20 == 0:
                donnees = {
                    "env_id": env_id,
                    "cars": [(c["x"], c["alive"]) for c in env.cars],
                    "obstacles": [(ob[0], ob[1]) for ob in env.obstacles],
                    "generation": generation,
                    "history": history_bests,
                    "record": all_time_best
                }
                # On utilise put_nowait pour ne pas bloquer le calcul si le tuyau est plein
                try:
                    queue_vers_affichage.put_nowait(donnees)
                except:
                    pass
            
        # Fin de génération
        best_score = -1
        best_index = 0
        for i, car in enumerate(env.cars):
            if car["time_alive"] > best_score:
                best_score = car["time_alive"]
                best_index = i
                
        history_bests.append(best_score)
        if best_score > all_time_best:
            all_time_best = best_score
            
        champion = copy.deepcopy(family[best_index])
        with open(fichier_env, "w") as f:
            json.dump(champion.export_json(), f, indent=4)
            
        generation += 1


def main_process_display():
    """Processus principal : gère l'affichage à 60 FPS."""
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(f"Car Dodge - {NUM_ENVS} CPUs à 100%")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    
    # Queue de communication (taille max pour éviter les fuites de mémoire)
    queue = mp.Queue(maxsize=100)
    
    processus = []
    for env_id in range(NUM_ENVS):
        p = mp.Process(target=worker_training_loop, args=(env_id, queue))
        p.daemon = True # Les processus enfants meurent si le principal meurt
        p.start()
        processus.append(p)
        
    dernieres_images = {i: None for i in range(NUM_ENVS)}
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            # Vider la queue et mettre à jour le dictionnaire
            while not queue.empty():
                try:
                    data = queue.get_nowait()
                    dernieres_images[data["env_id"]] = data
                except:
                    break
                
            # --- DESSIN DE L'ÉCRAN ---
            screen.fill((20, 20, 20))
            
            for env_id in range(NUM_ENVS):
                data = dernieres_images[env_id]
                offset_x = env_id * GAME_W 
                pygame.draw.line(screen, (100, 100, 100), (offset_x, 0), (offset_x, WINDOW_H), 2)
                
                if data is None:
                    continue # On n'a pas encore reçu d'infos de ce processus
                    
                # Dessiner obstacles
                for ob in data["obstacles"]:
                    pygame.draw.rect(screen, (220, 70, 70), (offset_x + ob[0], ob[1], OBSTACLE_W, OBSTACLE_H))
                
                # Dessiner voitures
                alives = 0
                for i, car in enumerate(data["cars"]):
                    car_x, is_alive = car
                    if is_alive:
                        alives += 1
                        if i == 0:
                            pygame.draw.rect(screen, (60, 200, 255), (offset_x + car_x, CAR_Y, CAR_W, CAR_H)) # Elite
                        else:
                            pygame.draw.rect(screen, (60, 200, 90), (offset_x + car_x, CAR_Y, CAR_W, CAR_H))  # Mutants
                
                # Texte haut
                txt = font.render(f"F{env_id+1} G{data['generation']} : {alives} en vie", True, (255, 255, 255))
                screen.blit(txt, (offset_x + 5, 10))
                
                # Panneau Statistiques
                pygame.draw.rect(screen, (35, 35, 45), (offset_x, GAME_H, GAME_W, 60))
                
                recent = data["history"][-10:] if len(data["history"]) > 0 else []
                avg_score = sum(recent) / len(recent) if recent else 0.0
                record = data["record"]
                
                txt_avg = font.render(f"Moy(10g): {avg_score:.1f}s", True, (150, 200, 255))
                txt_rec = font.render(f"Record: {record:.1f}s", True, (255, 215, 0))
                screen.blit(txt_avg, (offset_x + 5, GAME_H + 10))
                screen.blit(txt_rec, (offset_x + 5, GAME_H + 35))
            
            pygame.display.flip()
            clock.tick(60) # L'affichage tourne à 60 FPS, les calculs en arrière plan n'ont AUCUNE LIMITE.

    except KeyboardInterrupt:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    mp.freeze_support() 
    main_process_display()