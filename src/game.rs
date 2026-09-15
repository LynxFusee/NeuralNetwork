use rand::RngExt;

pub const GAME_W: f32 = 250.0;
pub const GAME_H: f32 = 450.0;
pub const CAR_W: f32 = 30.0;
pub const CAR_H: f32 = 45.0;
pub const CAR_Y: f32 = GAME_H - CAR_H - 10.0;
pub const CAR_SPEED: f32 = 5.0;

pub const OBSTACLE_W: f32 = 35.0;
pub const OBSTACLE_H: f32 = 35.0;
pub const OBSTACLE_START_SPEED: f32 = 4.0;
pub const OBSTACLE_SPEED_RAMP: f32 = 0.08;
pub const SPAWN_EVERY_MS_START: f32 = 800.0;
pub const SPAWN_EVERY_MS_MIN: f32 = 300.0;
pub const SPAWN_RAMP: f32 = 20.0;

#[derive(Clone)]
pub struct Car {
    pub x: f32,
    pub alive: bool,
    pub time_alive: f32,
}

#[derive(Clone)]
pub struct Obstacle {
    pub x: f32,
    pub y: f32,
    pub speed: f32,
}

pub struct CarDodgeEnv {
    pub cars: Vec<Car>,
    pub obstacles: Vec<Obstacle>,
    pub global_time: f32,
    pub spawn_timer: f32,
    pub spawn_interval: f32,
    pub obstacle_speed: f32,
    pub done: bool,
}

impl CarDodgeEnv {
    pub fn new(num_cars: usize) -> Self {
        let mut env = Self {
            cars: vec![],
            obstacles: vec![],
            global_time: 0.0,
            spawn_timer: 0.0,
            spawn_interval: SPAWN_EVERY_MS_START,
            obstacle_speed: OBSTACLE_START_SPEED,
            done: false,
        };
        env.reset(num_cars);
        env
    }

    pub fn reset(&mut self, num_cars: usize) {
        self.cars = vec![
            Car {
                x: GAME_W / 2.0 - CAR_W / 2.0,
                alive: true,
                time_alive: 0.0,
            };
            num_cars
        ];
        self.obstacles.clear();
        self.global_time = 0.0;
        self.spawn_timer = 0.0;
        self.spawn_interval = SPAWN_EVERY_MS_START;
        self.obstacle_speed = OBSTACLE_START_SPEED;
        self.done = false;
    }

    pub fn get_states(&self) -> Vec<Option<[f32; 8]>> {
        self.cars.iter().map(|car| {
            if !car.alive {
                return None;
            }
            
            let mut state = [0.0; 8];
            state[0] = car.x / GAME_W;
            state[1] = self.obstacle_speed / 20.0;

            let mut upcoming = self.obstacles.clone();
            upcoming.sort_by(|a, b| b.y.partial_cmp(&a.y).unwrap());

            for i in 0..3 {
                if i < upcoming.len() {
                    state[2 + i * 2] = upcoming[i].x / GAME_W;
                    state[3 + i * 2] = upcoming[i].y / GAME_H;
                } else {
                    state[2 + i * 2] = 0.0;
                    state[3 + i * 2] = -1.0;
                }
            }
            Some(state)
        }).collect()
    }

    pub fn step(&mut self, actions: &[u8], dt_ms: f32) {
        if self.done { return; }

        self.global_time += dt_ms / 1000.0;
        self.obstacle_speed = OBSTACLE_START_SPEED + self.global_time * OBSTACLE_SPEED_RAMP;
        self.spawn_interval = (SPAWN_EVERY_MS_START - self.global_time * SPAWN_RAMP).max(SPAWN_EVERY_MS_MIN);

        self.spawn_timer += dt_ms;
        if self.spawn_timer >= self.spawn_interval {
            self.spawn_timer = 0.0;
            let mut rng = rand::rng(); 
            let mut x = rng.random_range(0.0..(GAME_W - OBSTACLE_W));

            // On s'assure qu'un passage reste toujours ouvert par rapport au dernier obstacle
            if let Some(last_ob) = self.obstacles.last() {
                if (x - last_ob.x).abs() < OBSTACLE_W {
                    x = (x + OBSTACLE_W * 2.0) % (GAME_W - OBSTACLE_W);
                }
            }

            self.obstacles.push(Obstacle { x, y: -OBSTACLE_H, speed: self.obstacle_speed });
        }

        for ob in &mut self.obstacles {
            ob.y += ob.speed;
        }
        self.obstacles.retain(|ob| ob.y < GAME_H);

        let mut all_dead = true;
        for (i, car) in self.cars.iter_mut().enumerate() {
            if !car.alive { continue; }

            match actions[i] {
                1 => car.x -= CAR_SPEED, // Gauche
                2 => car.x += CAR_SPEED, // Droite
                _ => {}                  // Rester
            }
            car.x = car.x.clamp(0.0, GAME_W - CAR_W);
            car.time_alive += dt_ms / 1000.0;

            let mut collided = false;
            for ob in &self.obstacles {
                if car.x < ob.x + OBSTACLE_W &&
                   car.x + CAR_W > ob.x &&
                   CAR_Y < ob.y + OBSTACLE_H &&
                   CAR_Y + CAR_H > ob.y 
                {
                    collided = true;
                    break;
                }
            }

            if collided {
                car.alive = false;
            } else {
                all_dead = false;
            }
        }
        self.done = all_dead;
    }
}