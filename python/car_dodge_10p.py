import sys
import random
import pygame
import ai
import copy
import json
import os
import shutil
import multiprocessing as mp

# ==========================================
#         PARAMÈTRES GLOBAUX (i9-13900)
# ==========================================
NUM_ENVS = 24                  
COLS = 8                       
ROWS = (NUM_ENVS + COLS - 1) // COLS 

CARS_PER_ENV = 100             
MUTATION_RATE = 0.20           
FPS = 60                       

GAME_W, GAME_H = 250, 450      
WINDOW_W = GAME_W * COLS       
WINDOW_H = (GAME_H + 60) * ROWS 

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

# Paramètres IA
N_TRACKED_OBSTACLES = 3        
ENTRY_SIZE = 2 + (N_TRACKED_OBSTACLES * 2) # Position X, Vitesse globale + 3x(X,Y) = 8
HIDDEN_LAYER_SIZE = 16
EXIT_SIZE = 3 # Gauche, Droite, Rester
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
            
            # 1. Position IA & Vitesse
            state = [car["x"] / GAME_W, self.obstacle_speed / 20.0]
            
            # 2. Les 3 prochains obstacles
            upcoming = sorted(self.obstacles, key=lambda ob: ob[1])[:N_TRACKED_OBSTACLES]
            for ob in upcoming:
                state += [ob[0] / GAME_W, ob[1] / GAME_H]
                
            # 3. Remplissage si l'écran est vide
            while len(state) < ENTRY_SIZE:
                state += [0.0, -1.0] 
                
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


def worker_training_loop(env_id, queue_vers_affichage, pause_request_event, resume_event, save_dir):
    env = CarDodgeEnv(num_cars=CARS_PER_ENV)
    fichier_env = os.path.join(save_dir, f"ia_env_{env_id}.json")
    champion_loaded = False
    
    if os.path.exists(fichier_env):
        try:
            with open(fichier_env, "r") as f:
                data = json.load(f)
            # Vérifie que la sauvegarde correspond aux nouvelles dimensions
            if data["entry_size"] == ENTRY_SIZE and data["exit_size"] == EXIT_SIZE:
                champion = ai.Network(
                    layers_count=data["layers_count"], per_layer=data["per_layer"],
                    wheights=data["wheights"], bias=data["bias"],
                    entry_size=data["entry_size"], exit_size=data["exit_size"]
                )
                champion.createNetwork()
                champion_loaded = True
        except (json.JSONDecodeError, KeyError, FileNotFoundError):
            pass
            
    if not champion_loaded:
        champion = ai.Network(layers_count=1, per_layer=HIDDEN_LAYER_SIZE, wheights=None, bias=None, entry_size=ENTRY_SIZE, exit_size=EXIT_SIZE)
        champion.createNetwork()
        
    parents = [copy.deepcopy(champion) for _ in range(3)]
        
    global_generation = 1
    local_generation = 1
    tournaments = 0
    step_count = 0
    all_time_best = 0.0
    history_bests = []
    reached_current_milestone = False
    
    while True:
        family = []
        family.append(copy.deepcopy(parents[0])) 
        for _ in range(33):
            net = copy.deepcopy(parents[0])
            net.mutNetwork(MUTATION_RATE)
            family.append(net)
            
        for _ in range(33):
            net = copy.deepcopy(parents[1])
            net.mutNetwork(MUTATION_RATE)
            family.append(net)
            
        for _ in range(33):
            net = copy.deepcopy(parents[2])
            net.mutNetwork(MUTATION_RATE)
            family.append(net)
            
        states = env.reset()
        
        while not env.done:
            actions = []
            for i in range(CARS_PER_ENV):
                if env.cars[i]["alive"]:
                    state = states[i]
                    results = family[i].logic(state)
                    # Trouve l'index du neurone avec la valeur la plus haute (0=Gauche, 1=Droite, 2=Rester)
                    best_action_idx = results.index(max(results))
                    
                    if best_action_idx == 0: actions.append(ACTION_LEFT)
                    elif best_action_idx == 1: actions.append(ACTION_RIGHT)
                    else: actions.append(ACTION_STAY)
                else:
                    actions.append(ACTION_STAY)
                    
            states, _ = env.step(actions)
            step_count += 1
            
            if step_count % 30 == 0: 
                donnees = {
                    "type": "update", "env_id": env_id,
                    "cars": [(c["x"], c["alive"]) for c in env.cars],
                    "obstacles": [(ob[0], ob[1]) for ob in env.obstacles],
                    "global_gen": global_generation, "local_gen": local_generation,
                    "tournaments": tournaments, "history": history_bests,
                    "record": all_time_best
                }
                try: queue_vers_affichage.put_nowait(donnees)
                except: pass
            
        voitures_triees = sorted([(i, env.cars[i]["time_alive"]) for i in range(CARS_PER_ENV)], key=lambda x: x[1], reverse=True)
        best_score = voitures_triees[0][1]
        history_bests.append(best_score)
        
        parents[0] = copy.deepcopy(family[voitures_triees[0][0]])
        parents[1] = copy.deepcopy(family[voitures_triees[1][0]])
        parents[2] = copy.deepcopy(family[voitures_triees[2][0]])
        
        # Sauvegarde uniquement sur record ou toutes les 50 gen (Réduit les accès disque de 98%)
        if best_score > all_time_best:
            all_time_best = best_score
            with open(fichier_env, "w") as f: json.dump(parents[0].export_json(), f, indent=4)
        elif local_generation % 50 == 0:
            with open(fichier_env, "w") as f: json.dump(parents[0].export_json(), f, indent=4)
            
        if local_generation >= 500 and not reached_current_milestone:
            recent = history_bests[-100:]
            avg_score = sum(recent) / len(recent) if recent else 0.0
            queue_vers_affichage.put({"type": "checkpoint", "env_id": env_id, "avg": avg_score})
            reached_current_milestone = True

        if pause_request_event.is_set():
            queue_vers_affichage.put({"type": "paused", "env_id": env_id})
            resume_event.wait() 
            
            try:
                with open(fichier_env, "r") as f: data = json.load(f)
                nouveau_champion = ai.Network(
                    layers_count=data["layers_count"], per_layer=data["per_layer"],
                    wheights=data["wheights"], bias=data["bias"],
                    entry_size=data["entry_size"], exit_size=data["exit_size"]
                )
                nouveau_champion.createNetwork()
                parents = [copy.deepcopy(nouveau_champion) for _ in range(3)]
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                pass
            
            tournaments += 1
            local_generation = 0  
            reached_current_milestone = False
            history_bests = []

        global_generation += 1
        local_generation += 1

def main_process_display(save_dir):
    pygame.init()
    main_screen = pygame.display.set_mode((1280, 720), pygame.RESIZABLE)
    virtual_screen = pygame.Surface((WINDOW_W, WINDOW_H)) 
    pygame.display.set_caption(f"Car Dodge - {NUM_ENVS} CPUs | Vision Étendue")
    clock = pygame.time.Clock()
    font_small = pygame.font.SysFont(None, 20)
    font = pygame.font.SysFont(None, 24)
    
    queue = mp.Queue(maxsize=150)
    pause_request_event = mp.Event()
    resume_event = mp.Event()
    processus = []
    
    for env_id in range(NUM_ENVS):
        p = mp.Process(target=worker_training_loop, args=(env_id, queue, pause_request_event, resume_event, save_dir))
        p.daemon = True
        p.start()
        processus.append(p)
        
    dernieres_images = {i: None for i in range(NUM_ENVS)}
    checkpoints_atteints = {}
    paused_workers = set()
    tournament_in_progress = False
    
    try:
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            for _ in range(150):
                if queue.empty(): break
                try:
                    data = queue.get_nowait()
                    if data["type"] == "update": dernieres_images[data["env_id"]] = data
                    elif data["type"] == "checkpoint": checkpoints_atteints[data["env_id"]] = data["avg"]
                    elif data["type"] == "paused": paused_workers.add(data["env_id"])
                except: break
                    
            if len(checkpoints_atteints) == NUM_ENVS and not tournament_in_progress:
                print("\n=== ✨ TOUT LE MONDE A FRANCHI LE CAP ! ✨ ===")
                tournament_in_progress = True
                resume_event.clear()
                pause_request_event.set()

            if tournament_in_progress and len(paused_workers) == NUM_ENVS:
                top_4 = sorted(checkpoints_atteints.items(), key=lambda x: x[1], reverse=True)[:4]
                print(f"🥇 1er: IA n°{top_4[0][0]+1} (Moyenne: {top_4[0][1]:.1f}s)")
                print(f"🥈 2e : IA n°{top_4[1][0]+1} (Moyenne: {top_4[1][1]:.1f}s)")
                print(f"🥉 3e : IA n°{top_4[2][0]+1} (Moyenne: {top_4[2][1]:.1f}s)")
                print(f"🏅 4e : IA n°{top_4[3][0]+1} (Moyenne: {top_4[3][1]:.1f}s)")
                
                chunk_size = max(1, NUM_ENVS // 4)
                for i in range(NUM_ENVS):
                    target = os.path.join(save_dir, f"ia_env_{i}.json")
                    source_ia_rank = min(i // chunk_size, 3) 
                    source_ia_id = top_4[source_ia_rank][0]
                    source_file = os.path.join(save_dir, f"ia_env_{source_ia_id}.json")
                    if target != source_file: shutil.copyfile(source_file, target)
                        
                print("==> IA distribuées (4 Vainqueurs). Reprise immédiate !\n")
                checkpoints_atteints.clear()
                paused_workers.clear()
                tournament_in_progress = False
                pause_request_event.clear()
                resume_event.set() 
                
            virtual_screen.fill((20, 20, 20))
            
            for env_id in range(NUM_ENVS):
                data = dernieres_images[env_id]
                col, row = env_id % COLS, env_id // COLS
                offset_x, offset_y = col * GAME_W, row * (GAME_H + 60)
                
                pygame.draw.line(virtual_screen, (100, 100, 100), (offset_x, offset_y), (offset_x, offset_y + GAME_H + 60), 2)
                pygame.draw.line(virtual_screen, (100, 100, 100), (offset_x, offset_y), (offset_x + GAME_W, offset_y), 2)
                
                if data is None: continue 
                    
                for ob in data["obstacles"]:
                    pygame.draw.rect(virtual_screen, (220, 70, 70), (offset_x + ob[0], offset_y + ob[1], OBSTACLE_W, OBSTACLE_H))
                
                alives = 0
                for i, car in enumerate(data["cars"]):
                    car_x, is_alive = car
                    if is_alive:
                        alives += 1
                        color = (60, 200, 255) if i == 0 else (60, 200, 90)
                        pygame.draw.rect(virtual_screen, color, (offset_x + car_x, offset_y + CAR_Y, CAR_W, CAR_H))
                
                txt_status = f"F{env_id+1} | {alives} en vie"
                if pause_request_event.is_set() and env_id in paused_workers: txt_status += " (PAUSE)"
                
                label_1 = font.render(txt_status, True, (255, 255, 255))
                label_2 = font_small.render(f"Tournois joués: {data['tournaments']}", True, (255, 200, 100))
                label_3 = font_small.render(f"Gen globale: {data['global_gen']}", True, (150, 200, 255))
                label_4 = font_small.render(f"Gen locale: {data['local_gen']}", True, (150, 255, 150))
                
                bg_rect = pygame.Surface((GAME_W - 2, 75))
                bg_rect.set_alpha(150)
                bg_rect.fill((0, 0, 0))
                virtual_screen.blit(bg_rect, (offset_x + 1, offset_y))
                virtual_screen.blit(label_1, (offset_x + 5, offset_y + 5))
                virtual_screen.blit(label_2, (offset_x + 5, offset_y + 25))
                virtual_screen.blit(label_3, (offset_x + 5, offset_y + 40))
                virtual_screen.blit(label_4, (offset_x + 5, offset_y + 55))
                
                pygame.draw.rect(virtual_screen, (35, 35, 45), (offset_x, offset_y + GAME_H, GAME_W, 60))
                recent = data["history"][-100:] if len(data["history"]) > 0 else []
                avg_score = sum(recent) / len(recent) if recent else 0.0
                record = data["record"]
                
                txt_avg = font.render(f"Moy(100g): {avg_score:.1f}s", True, (150, 200, 255))
                txt_rec = font.render(f"Record: {record:.1f}s", True, (255, 215, 0))
                virtual_screen.blit(txt_avg, (offset_x + 5, offset_y + GAME_H + 10))
                virtual_screen.blit(txt_rec, (offset_x + 5, offset_y + GAME_H + 35))
            
            scaled_screen = pygame.transform.scale(virtual_screen, main_screen.get_size())
            main_screen.blit(scaled_screen, (0, 0))
            pygame.display.flip()
            clock.tick(60) 

    except KeyboardInterrupt:
        pygame.quit()
        sys.exit()

if __name__ == "__main__":
    mp.freeze_support() 
    mp.set_start_method('spawn', force=True) 
    
    if len(sys.argv) < 2:
        print("❌ Erreur : Vous devez spécifier le nom du dossier de sauvegarde (ex: nouveau_test).")
        sys.exit(1)
        
    dossier_choisi = sys.argv[1]
    os.makedirs(dossier_choisi, exist_ok=True)
    print(f"📁 Dossier de travail : {dossier_choisi}")
    print(f"🚀 Lancement de l'entraînement sur {NUM_ENVS} cœurs !")
    main_process_display(dossier_choisi)n le programme plantera en 