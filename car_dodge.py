import sys
import random
import pygame
import ai
import copy
import json
import os

SCREEN_W, SCREEN_H = 480, 640
CAR_W, CAR_H = 50, 80
CAR_Y = SCREEN_H - CAR_H - 20
CAR_SPEED = 8

OBSTACLE_W, OBSTACLE_H = 50, 50
OBSTACLE_START_SPEED = 5
OBSTACLE_SPEED_RAMP = 0.09
SPAWN_EVERY_MS_START = 900
SPAWN_EVERY_MS_MIN = 350
SPAWN_RAMP = 20

N_TRACKED_OBSTACLES = 3
FPS = 60

ACTION_STAY, ACTION_LEFT, ACTION_RIGHT = 0, 1, 2


class CarDodgeEnv:
    """Environnement modifié pour gérer N voitures simultanément."""
    def __init__(self, num_cars=10):
        self.num_cars = num_cars
        self.reset()

    def reset(self):
        # Chaque voiture a sa propre position, son état (en vie) et son temps de survie
        self.cars = [{"x": SCREEN_W / 2 - CAR_W / 2, "alive": True, "time_alive": 0.0} for _ in range(self.num_cars)]
        
        self.obstacles = []
        self.global_time = 0.0
        self.spawn_timer = 0.0
        self.spawn_interval = SPAWN_EVERY_MS_START
        self.obstacle_speed = OBSTACLE_START_SPEED
        self.done = False
        return self.get_states()

    def get_states(self):
        """Génère la liste des 'états' (la vue) pour chaque voiture en vie."""
        states = []
        for car in self.cars:
            if not car["alive"]:
                states.append(None) # La voiture est morte, pas besoin d'état
                continue
            
            state = [car["x"] / SCREEN_W]
            upcoming = sorted(self.obstacles, key=lambda ob: ob[1])[:N_TRACKED_OBSTACLES]
            for ob in upcoming:
                state += [ob[0] / SCREEN_W, ob[1] / SCREEN_H, ob[2] / 20]
            while len(state) < 1 + N_TRACKED_OBSTACLES * 3:
                state += [0.0, -1.0, 0.0]
            states.append(state)
        return states

    def step(self, actions, dt_ms=1000 / FPS):
        if self.done:
            raise RuntimeError("Épisode terminé, appelez reset().")

        # 1. Gérer la difficulté globale
        self.global_time += dt_ms / 1000
        self.obstacle_speed = OBSTACLE_START_SPEED + self.global_time * OBSTACLE_SPEED_RAMP
        self.spawn_interval = max(SPAWN_EVERY_MS_MIN, SPAWN_EVERY_MS_START - self.global_time * SPAWN_RAMP)

        # 2. Faire apparaître les obstacles (Communs à toutes les voitures)
        self.spawn_timer += dt_ms
        if self.spawn_timer >= self.spawn_interval:
            self.spawn_timer = 0
            x = random.randint(0, SCREEN_W - OBSTACLE_W)
            self.obstacles.append([x, -OBSTACLE_H, self.obstacle_speed])

        for ob in self.obstacles:
            ob[1] += ob[2]
        self.obstacles = [ob for ob in self.obstacles if ob[1] < SCREEN_H]

        # 3. Mettre à jour chaque voiture
        all_dead = True
        for i, car in enumerate(self.cars):
            if not car["alive"]:
                continue
            
            act = actions[i]
            if act == ACTION_LEFT:
                car["x"] -= CAR_SPEED
            elif act == ACTION_RIGHT:
                car["x"] += CAR_SPEED
            car["x"] = max(0, min(SCREEN_W - CAR_W, car["x"]))

            car["time_alive"] += dt_ms / 1000

            # Collision pour cette voiture
            car_rect = pygame.Rect(car["x"], CAR_Y, CAR_W, CAR_H)
            collided = any(car_rect.colliderect(pygame.Rect(ob[0], ob[1], OBSTACLE_W, OBSTACLE_H)) for ob in self.obstacles)
            
            if collided:
                car["alive"] = False
            else:
                all_dead = False # Il reste au moins un survivant

        self.done = all_dead
        return self.get_states(), self.done


def train_ai():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_W, SCREEN_H))
    pygame.display.set_caption("Car Dodge - Entraînement IA Simultané")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 28)
    
    num_ais = 20
    env = CarDodgeEnv(num_cars=num_ais)
    
    FICHIER_SAUVEGARDE = "ia.json"
    
    # --- CHARGEMENT OU CRÉATION DU MODÈLE ---
    if os.path.exists(FICHIER_SAUVEGARDE):
        print("💾 Fichier de sauvegarde trouvé ! Chargement de l'IA...")
        with open(FICHIER_SAUVEGARDE, "r") as f:
            data = json.load(f)
            
        best_net = ai.Network(
            layers_count=data["layers_count"],
            per_layer=data["per_layer"],
            wheights=data["wheights"],
            bias=data["bias"],
            entry_size=data["entry_size"],
            exit_size=data["exit_size"]
        )
        best_net.createNetwork()
    else:
        print("🌱 Aucune sauvegarde trouvée. Création d'une nouvelle IA de zéro...")
        best_net = ai.Network(layers_count=5, per_layer=50, wheights=None, bias=None, entry_size=10, exit_size=1)
        best_net.createNetwork()
    
    generation = 1
    
    while True:
        networks = []
        for i in range(num_ais):
            new_net = copy.deepcopy(best_net)
            if i > 0:
                new_net.mutNetwork(i / num_ais)
            networks.append(new_net)
        
        states = env.reset()
        done = False
        
        # --- BOUCLE DE JEU ---
        while not done:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit()
                    sys.exit()
            
            actions = []
            for i in range(num_ais):
                if env.cars[i]["alive"]:
                    state = states[i]
                    result = networks[i].logic(state)[0]
                    if result <= -0.33:
                        actions.append(ACTION_LEFT)
                    elif result >= 0.33:
                        actions.append(ACTION_RIGHT)
                    else:
                        actions.append(ACTION_STAY)
                else:
                    actions.append(ACTION_STAY)
                    
            states, done = env.step(actions)
            
            screen.fill((25, 25, 35))
            for ob in env.obstacles:
                pygame.draw.rect(screen, (220, 70, 70), (ob[0], ob[1], OBSTACLE_W, OBSTACLE_H))
                
            alives = 0
            for i, car in enumerate(env.cars):
                if car["alive"] and i > 0:
                    alives += 1
                    pygame.draw.rect(screen, (60, 200, 90), (car["x"], CAR_Y, CAR_W, CAR_H))
            
            if env.cars[0]["alive"]:
                alives += 1
                pygame.draw.rect(screen, (60, 200, 255), (env.cars[0]["x"], CAR_Y, CAR_W, CAR_H))
            
            text = font.render(f"Gen: {generation} | En vie: {alives}/{num_ais}", True, (255, 255, 255))
            screen.blit(text, (10, 10))
            
            pygame.display.flip()
            clock.tick(FPS*2)
            
        # --- FIN DE GÉNÉRATION ET SAUVEGARDE ---
        best_score = -1
        best_index = 0
        for i, car in enumerate(env.cars):
            if car["time_alive"] > best_score:
                best_score = car["time_alive"]
                best_index = i
        
        print(f"Génération {generation} terminée. Meilleur temps: {best_score:.1f}s (IA n°{best_index})")
        
        # 1. Mettre à jour l'IA de référence
        best_net = copy.deepcopy(networks[best_index])
        
        # 2. Sauvegarder dans le fichier JSON
        with open(FICHIER_SAUVEGARDE, "w") as f:
            json.dump(best_net.export_json(), f, indent=4)

        generation += 1


if __name__ == "__main__":
    train_ai()