use macroquad::prelude::*;
use std::fs;
use std::env;

#[path = "../ia.rs"]
mod ia;
#[path = "../game.rs"]
mod game;

use ia::{Network, ModelSave};
use game::{CarDodgeEnv, GAME_W, GAME_H, CAR_W, CAR_H, CAR_Y, OBSTACLE_W, OBSTACLE_H};

#[macroquad::main("Car Dodge - Mode IA & Visualisation")]
async fn main() {
    request_new_screen_size(900.0, 700.0);

    let args: Vec<String> = env::args().collect();
    let save_file = if args.len() > 1 {
        args[1].clone()
    } else {
        String::from("best_model.json")
    };

    let json_data = fs::read_to_string(&save_file).expect("Impossible de lire le fichier JSON du modèle.");
    let ai_brain: Network = if let Ok(save_data) = serde_json::from_str::<ModelSave>(&json_data) {
        save_data.network
    } else {
        serde_json::from_str(&json_data).expect("Erreur lors du décodage du réseau de l'ancien format.")
    };

    let mut env = CarDodgeEnv::new(1);

    loop {
        clear_background(BLACK);

        let mut activations: Vec<Vec<f32>> = vec![vec![0.0; 8]];

        if !env.done {
            let states = env.get_states();
            let mut action = 0;

            if let Some(state_array) = states[0] {
                let current_states = state_array.to_vec();
                activations = ai_brain.forward_all(current_states);
                let outputs = activations.last().unwrap().clone();

                let mut max_val = outputs[2];
                if outputs[0] > max_val { max_val = outputs[0]; action = 1; }
                if outputs[1] > max_val { action = 2; }
            }

            env.step(&[action], get_frame_time() * 1000.0);
        } else {
            env.reset(1);
        }

        let game_screen_w = 400.0;
        let scale_x = game_screen_w / GAME_W;
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
            BLUE,
        );

        let text = format!("Survie IA : {:.2}s", car.time_alive);
        draw_text(&text, 20.0, 40.0, 25.0, WHITE);

        draw_line(game_screen_w, 0.0, game_screen_w, screen_height(), 2.0, DARKGRAY);

        let net_origin_x = 440.0;
        let net_origin_y = 50.0;
        let num_layers = activations.len();
        let layer_spacing = 440.0 / ((num_layers.max(2) - 1) as f32);

        let mut layer_coords: Vec<Vec<(f32, f32, f32)>> = vec![];

        for (l_idx, layer_vals) in activations.iter().enumerate() {
            let x = net_origin_x + (l_idx as f32) * layer_spacing;
            let num_nodes = layer_vals.len();
            let total_height = 600.0;
            let node_spacing = (total_height / (num_nodes as f32 + 1.0)).min(35.0);
            let start_y = net_origin_y + (total_height - (num_nodes as f32 - 1.0) * node_spacing) / 2.0;

            let mut node_list = vec![];
            for (n_idx, &val) in layer_vals.iter().enumerate() {
                let y = start_y + (n_idx as f32) * node_spacing;
                node_list.push((x, y, val));
            }
            layer_coords.push(node_list);
        }

        for l in 0..(layer_coords.len() - 1) {
            let current = &layer_coords[l];
            let next = &layer_coords[l + 1];
            for c in current {
                for n in next {
                    draw_line(c.0, c.1, n.0, n.1, 0.5, DARKGRAY);
                }
            }
        }

        for layer in &layer_coords {
            for node in layer {
                let activation = node.2.clamp(-1.0, 1.0);
                let color = if activation > 0.0 {
                    Color::new(1.0, 1.0 - activation * 0.8, 1.0 - activation * 0.8, 1.0)
                } else {
                    Color::new(1.0 + activation * 0.8, 1.0 + activation * 0.8, 1.0, 1.0)
                };
                draw_circle(node.0, node.1, 6.0, color);
                draw_circle_lines(node.0, node.1, 6.0, 1.0, WHITE);
            }
        }

        draw_text("Cerveau de l'IA (Temps réel)", net_origin_x, 30.0, 18.0, WHITE);

        next_frame().await;
    }
}