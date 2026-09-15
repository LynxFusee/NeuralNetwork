use macroquad::prelude::*;

#[path = "../game.rs"]
mod game;
use game::{CarDodgeEnv, GAME_W, GAME_H, CAR_W, CAR_H, CAR_Y, CAR_SPEED, OBSTACLE_W, OBSTACLE_H};

#[macroquad::main("Car Dodge - Mode Joueur")]
async fn main() {
    request_new_screen_size(400.0, 700.0);

    let mut env = CarDodgeEnv::new(1);
    let mut game_over = false;

    loop {
        clear_background(BLACK);

        if !game_over {
            let mut action = 0; 
            if is_key_down(KeyCode::Left) || is_key_down(KeyCode::A) {
                action = 1;
            } else if is_key_down(KeyCode::Right) || is_key_down(KeyCode::D) {
                action = 2;
            }

            env.step(&[action], get_frame_time() * 1000.0);
            game_over = env.done;
        } else {
            if is_key_pressed(KeyCode::Space) {
                env.reset(1);
                game_over = false;
            }
        }

        let scale_x = screen_width() / GAME_W;
        let scale_y = screen_height() / GAME_H;

        for ob in &env.obstacles {
            draw_rectangle(
                ob.x * scale_x,
                ob.y * scale_y,
                OBSTACLE_W * scale_x,
                OBSTACLE_H * scale_y,
                RED,
            );
        }

        let car = &env.cars[0];
        draw_rectangle(
            car.x * scale_x,
            CAR_Y * scale_y,
            CAR_W * scale_x,
            CAR_H * scale_y,
            GREEN,
        );

        let text = format!("Temps : {:.2}s", car.time_alive);
        draw_text(&text, 20.0, 40.0, 30.0, WHITE);

        if game_over {
            draw_text("GAME OVER", screen_width() / 2.0 - 100.0, screen_height() / 2.0, 40.0, RED);
            draw_text("Espace pour rejouer", screen_width() / 2.0 - 110.0, screen_height() / 2.0 + 50.0, 20.0, WHITE);
        }

        next_frame().await;
    }
}