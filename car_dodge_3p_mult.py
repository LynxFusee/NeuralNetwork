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
#         PARAMÈTRES GLOBAUX
# ==========================================
NUM_ENVS = 8                   
CARS_PER_ENV = 100             
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

# L'IA ne regarde que 1 seul obstacle (sa position + X et Y de l'obstacle = 3 entrées)
N_TRACKED_OBSTACLES = 1        
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
            
            # Les 3 variables d'entrée : Position IA (X), Position Obstacle (X), Position Obstacle (Y)
            state = [car["x"] / GAME_W]
            upcoming = sorted(self.obstacles, key=lambda ob: ob[1])
            if upcoming:
                state += [upcoming[0][0] / GAME_W, upcoming[0][1] / GAME_H]
            else:
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
    
    # --- FILET DE SÉCURITÉ ANTI-CORRUPTION AU DÉMARRAGE ---
    if os.path.exists(fichier_env):
        try:
            with open(fichier_env, "r") as f:
                data = json.load(f)
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
        champion = ai.Network(layers_count=1, per_layer=6, wheights=None, bias=None, entry_size=3, exit_size=1)
        champion.createNetwork()
        
    # On initialise la liste des 3 parents
    parents = [copy.deepcopy(champion) for _ in range(3)]
        
    global_generation = 1
    local_generation = 1
    tournaments = 0
    
    step_count = 0
    all_time_best = 0.0
    history_bests = []
    
    reached_current_milestone = False
    
    while True:
        # --- NOUVELLE LOGIQUE DE POPULATION (100 IA) ---
        family = []
        
        # 1. Héritage du Parent n°1 (33 IA)
        family.append(copy.deepcopy(parents[0])) # Le clone parfait du Top 1
        for _ in range(32):
            net = copy.deepcopy(parents[0])
            net.mutNetwork(MUTATION_RATE)
            family.append(net)
            
        # 2. Héritage du Parent n°2 (33 IA)
        for _ in range(33):
            net = copy.deepcopy(parents[1])
            net.mutNetwork(MUTATION_RATE)
            family.append(net)
            
        # 3. Héritage du Parent n°3 (33 IA)
        for _ in range(33):
            net = copy.deepcopy(parents[2])
            net.mutNetwork(MUTATION_RATE)
            family.append(net)
            
        # 4. Sang neuf (1 IA 100% aléatoire)
        random_net = ai.Network(layers_count=1, per_layer=6, wheights=None, bias=None, entry_size=3, exit_size=1)
        random_net.createNetwork()
        family.append(random_net)
            
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
            
            # Envoi visuel
            if step_count % 20 == 0:
                donnees = {
                    "type": "update",
                    "env_id": env_id,
                    "cars": [(c["x"], c["alive"]) for c in env.cars],
                    "obstacles": [(ob[0], ob[1]) for ob in env.obstacles],
                    "global_gen": global_generation,
                    "local_gen": local_generation,
                    "tournaments": tournaments,
                    "history": history_bests,
                    "record": all_time_best
                }
                try: queue_vers_affichage.put_nowait(donnees)
                except: pass
            
        # --- Fin de génération : Sélection des 3 meilleurs ---
        voitures_triees = sorted(
            [(i, env.cars[i]["time_alive"]) for i in range(CARS_PER_ENV)], 
            key=lambda x: x[1], 
            reverse=True
        )
        
        best_score = voitures_triees[0][1]
        history_bests.append(best_score)
        
        if best_score > all_time_best:
            all_time_best = best_score
            
        # Mise à jour des 3 parents pour la génération suivante
        parents[0] = copy.deepcopy(family[voitures_triees[0][0]])
        parents[1] = copy.deepcopy(family[voitures_triees[1][0]])
        parents[2] = copy.deepcopy(family[voitures_triees[2][0]])
        
        # Le champion absolu (Parent 1) est sauvegardé pour le Tournoi
        with open(fichier_env, "w") as f:
            json.dump(parents[0].export_json(), f, indent=4)
            
        # --- LOGIQUE DU TOURNOI ---
        
        if local_generation >= 500 and not reached_current_milestone:
            recent = history_bests[-100:]
            avg_score = sum(recent) / len(recent) if recent else 0.0
            queue_vers_affichage.put({"type": "checkpoint", "env_id": env_id, "avg": avg_score})
            reached_current_milestone = True

        if pause_request_event.is_set():
            queue_vers_affichage.put({"type": "paused", "env_id": env_id})
            resume_event.wait() 
            
            try:
                with open(fichier_env, "r") as f:
                    data = json.load(f)
                nouveau_champion = ai.Network(
                    layers_count=data["layers_count"], per_layer=data["per_layer"],
                    wheights=data["wheights"], bias=data["bias"],
                    entry_size=data["entry_size"], exit_size=data["exit_size"]
                )
                nouveau_champion.createNetwork()
                
                # Après un tournoi, les 3 parents repartent du champion imposé
                parents = [copy.deepcopy(nouveau_champion) for _ in range(3)]
                
            except (json.JSONDecodeError, KeyError, FileNotFoundError):
                pass
            
            # Réinitialisation des stats pour le nouveau tournoi
            tournaments += 1
            local_generation = 0  
            reached_current_milestone = False
            history_bests = []

        global_generation += 1
        local_generation += 1


def main_process_display(save_dir):
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_W, WINDOW_H))
    pygame.display.set_caption(f"Car Dodge - {NUM_ENVS} CPUs | Gen: 500 Cross-Pollination")
    clock = pygame.time.Clock()
    font_small = pygame.font.SysFont(None, 20)
    font = pygame.font.SysFont(None, 24)
    
    queue = mp.Queue(maxsize=100)
    
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
            
            # --- LECTURE SÉCURISÉE DE LA QUEUE --- (Anti-Freeze)
            for _ in range(100):
                if queue.empty():
                    break
                try:
                    data = queue.get_nowait()
                    if data["type"] == "update":
                        dernieres_images[data["env_id"]] = data
                    elif data["type"] == "checkpoint":
                        checkpoints_atteints[data["env_id"]] = data["avg"]
                    elif data["type"] == "paused":
                        paused_workers.add(data["env_id"])
                except:
                    break
                    
            # --- DÉCLENCHER LE TOURNOI ---
            if len(checkpoints_atteints) == NUM_ENVS and not tournament_in_progress:
                print("\n=== ✨ TOUT LE MONDE A FRANCHI LE CAP ! ✨ ===")
                tournament_in_progress = True
                resume_event.clear()
                pause_request_event.set()

            # --- PROCESSUS DU TOURNOI ---
            if tournament_in_progress and len(paused_workers) == NUM_ENVS:
                top_2 = sorted(checkpoints_atteints.items(), key=lambda x: x[1], reverse=True)[:2]
                ia_rank_1, score_1 = top_2[0]
                ia_rank_2, score_2 = top_2[1]
                
                print(f"🥇 IA n°{ia_rank_1+1} sélectionnée (Moyenne: {score_1:.1f}s)")
                print(f"🥈 IA n°{ia_rank_2+1} sélectionnée (Moyenne: {score_2:.1f}s)")
                
                file_rank_1 = os.path.join(save_dir, f"ia_env_{ia_rank_1}.json")
                file_rank_2 = os.path.join(save_dir, f"ia_env_{ia_rank_2}.json")
                
                moitie = NUM_ENVS // 2
                for i in range(NUM_ENVS):
                    target = os.path.join(save_dir, f"ia_env_{i}.json")
                    source_file = file_rank_1 if i < moitie else file_rank_2
                    if target != source_file:
                        shutil.copyfile(source_file, target) # Utilise copyfile pour WSL/Windows
                        
                print("==> IA distribuées (Moitié 🥇 | Moitié 🥈). Reprise immédiate !\n")
                
                checkpoints_atteints.clear()
                paused_workers.clear()
                tournament_in_progress = False
                
                pause_request_event.clear()
                resume_event.set() 
                
            # --- DESSIN DE L'ÉCRAN ---
            screen.fill((20, 20, 20))
            
            for env_id in range(NUM_ENVS):
                data = dernieres_images[env_id]
                offset_x = env_id * GAME_W 
                pygame.draw.line(screen, (100, 100, 100), (offset_x, 0), (offset_x, WINDOW_H), 2)
                
                if data is None:
                    continue 
                    
                for ob in data["obstacles"]:
                    pygame.draw.rect(screen, (220, 70, 70), (offset_x + ob[0], ob[1], OBSTACLE_W, OBSTACLE_H))
                
                alives = 0
                for i, car in enumerate(data["cars"]):
                    car_x, is_alive = car
                    if is_alive:
                        alives += 1
                        if i == 0:
                            pygame.draw.rect(screen, (60, 200, 255), (offset_x + car_x, CAR_Y, CAR_W, CAR_H)) 
                        else:
                            pygame.draw.rect(screen, (60, 200, 90), (offset_x + car_x, CAR_Y, CAR_W, CAR_H))
                
                # --- AFFICHAGE DES TEXTES DÉTAILLÉS EN HAUT ---
                txt_status = f"F{env_id+1} | {alives} en vie"
                if pause_request_event.is_set() and env_id in paused_workers:
                    txt_status += " (PAUSE)"
                
                label_1 = font.render(txt_status, True, (255, 255, 255))
                label_2 = font_small.render(f"Tournois joués: {data['tournaments']}", True, (255, 200, 100))
                label_3 = font_small.render(f"Gen globale: {data['global_gen']}", True, (150, 200, 255))
                label_4 = font_small.render(f"Gen locale: {data['local_gen']}", True, (150, 255, 150))
                
                # Fond semi-transparent pour que le texte reste lisible sur les obstacles
                bg_rect = pygame.Surface((GAME_W - 2, 75))
                bg_rect.set_alpha(150)
                bg_rect.fill((0, 0, 0))
                screen.blit(bg_rect, (offset_x + 1, 0))
                
                screen.blit(label_1, (offset_x + 5, 5))
                screen.blit(label_2, (offset_x + 5, 25))
                screen.blit(label_3, (offset_x + 5, 40))
                screen.blit(label_4, (offset_x + 5, 55))
                
                # Panneau Statistiques du bas
                pygame.draw.rect(screen, (35, 35, 45), (offset_x, GAME_H, GAME_W, 60))
                recent = data["history"][-100:] if len(data["history"]) > 0 else []
                avg_score = sum(recent) / len(recent) if recent else 0.0
                record = data["record"]
                
                txt_avg = font.render(f"Moy(100g): {avg_score:.1f}s", True, (150, 200, 255))
                txt_rec = font.render(f"Record: {record:.1f}s", True, (255, 215, 0))
                screen.blit(txt_avg, (offset_x + 5, GAME_H + 10))
                screen.blit(txt_rec, (offset_x + 5, GAME_H + 35))
            
            pygame.display.flip()
            clock.tick(60) 

    except KeyboardInterrupt:
        pygame.quit()
        sys.exit()


if __name__ == "__main__":
    mp.freeze_support() 
    # Force le contexte pour éviter le crash "context already set"
    mp.set_start_method('spawn', force=True) 
    
    if len(sys.argv) < 2:
        print("❌ Erreur : Vous devez spécifier le nom du dossier de sauvegarde.")
        print("💡 Exemple : python car_dodge_mult.py mon_test_ia_1")
        sys.exit(1)
        
    dossier_choisi = sys.argv[1]
    os.makedirs(dossier_choisi, exist_ok=True)
    
    print(f"📁 Dossier de travail : {dossier_choisi}")
    main_process_display(dossier_choisi)