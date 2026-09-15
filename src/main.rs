mod ia;
mod manager;
mod game;

use game::CarDodgeEnv;
use manager::Population;
use ia::{Network, ModelSave};
use std::fs;
use std::path::Path;
use std::env;
use rayon::prelude::*; 

fn main() {
    let args: Vec<String> = env::args().collect();
    let save_file = if args.len() > 1 {
        args[1].clone()
    } else {
        String::from("best_model.json")
    };

    println!("Démarrage de l'entraînement continu (Mode Multithread) !");
    println!("Fichier de sauvegarde cible : {}", save_file);
    println!("L'IA va s'entraîner à l'infini. Appuie sur Ctrl+C pour arrêter.");

    let pop_size = 100;
    let per_layer = vec![16, 8, 3]; 
    let num_games = 100;

    let mut initial_agents = None;
    let mut best_all_time_score = 0.0;
    let mut best_all_time_max = 0.0;
    let mut best_all_time_network = Network::new(per_layer.clone(), 8, None, None);

    if Path::new(&save_file).exists() {
        println!("Sauvegarde trouvée ! Chargement de {}...", save_file);
        if let Ok(json_data) = fs::read_to_string(&save_file) {
            if let Ok(save_data) = serde_json::from_str::<ModelSave>(&json_data) {
                initial_agents = Some(vec![save_data.network.clone(); pop_size]);
                best_all_time_score = save_data.average_score;
                best_all_time_max = save_data.max_score;
                best_all_time_network = save_data.network;
                println!("Modèle chargé avec succès (Record : {:.2}s / Max : {:.2}s) !", best_all_time_score, best_all_time_max);
            } else if let Ok(champion) = serde_json::from_str::<Network>(&json_data) {
                initial_agents = Some(vec![champion.clone(); pop_size]);
                best_all_time_network = champion;
                println!("Ancien format de modèle chargé et converti avec succès !");
            }
        }
    } else {
        println!("Aucune sauvegarde trouvée pour '{}'. Création d'une nouvelle lignée.", save_file);
    }

    let mut population = Population::new(pop_size, per_layer.clone(), 8, initial_agents);
    if best_all_time_score == 0.0 {
        best_all_time_network = population.agents[0].clone();
    }
    
    let mut score_at_last_save = best_all_time_score;

    loop {
        let current_gen = population.generation;

        let (total_scores, max_scores) = (0..num_games)
            .into_par_iter() 
            .map(|_| {
                let mut env = CarDodgeEnv::new(pop_size);

                while !env.done {
                    let states = env.get_states();
                    let mut actions: Vec<u8> = vec![0; pop_size];

                    for i in 0..pop_size {
                        if let Some(state_array) = states[i] {
                            let entries = state_array.to_vec();
                            let outputs = population.agents[i].logic(entries);

                            let mut best_action = 0; 
                            let mut max_val = outputs[2]; 

                            if outputs[0] > max_val { max_val = outputs[0]; best_action = 1; } 
                            if outputs[1] > max_val { best_action = 2; } 

                            actions[i] = best_action;
                        }
                    }
                    env.step(&actions, 16.0);
                }

                let scores: Vec<f32> = env.cars.iter().map(|car| car.time_alive).collect();
                (scores.clone(), scores)
            })
            .reduce(
                || (vec![0.0; pop_size], vec![0.0; pop_size]),
                |mut acc, curr| {
                    for i in 0..pop_size {
                        acc.0[i] += curr.0[i];
                        if curr.1[i] > acc.1[i] {
                            acc.1[i] = curr.1[i];
                        }
                    }
                    acc
                },
            );

        let mut scored_agents: Vec<(f32, f32, Network)> = vec![];
        for i in 0..pop_size {
            let average_score = total_scores[i] / (num_games as f32);
            let max_score = max_scores[i];
            scored_agents.push((average_score, max_score, population.agents[i].clone()));
        }

        scored_agents.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        println!(
            "Génération {} - Meilleur temps MOYEN : {:.2}s | Meilleur essai unique : {:.2}s", 
            current_gen, 
            scored_agents[0].0,
            scored_agents[0].1
        );

        if scored_agents[0].0 > best_all_time_score {
            best_all_time_score = scored_agents[0].0;
            best_all_time_max = scored_agents[0].1;
            best_all_time_network = scored_agents[0].2.clone();
        }

        if current_gen > 0 && current_gen % 100 == 0 {
            if best_all_time_score > score_at_last_save {
                println!("--> 💾 Nouveau record absolu ! Sauvegarde automatique (Génération {})...", current_gen);
                let save_data = ModelSave {
                    network: best_all_time_network.clone(),
                    average_score: best_all_time_score,
                    max_score: best_all_time_max,
                };
                if let Ok(json_data) = serde_json::to_string_pretty(&save_data) {
                    if fs::write(&save_file, json_data).is_ok() {
                        println!("--> ✅ Modèle sauvegardé dans '{}' (Record moyen : {:.2}s) !", save_file, best_all_time_score);
                    } else {
                        println!("--> ❌ Erreur : Impossible d'écrire sur le disque.");
                    }
                }
                score_at_last_save = best_all_time_score;
            } else {
                println!("--> 🔄 Aucun progrès sur 100 générations. Restauration de la dernière sauvegarde...");
                if let Ok(json_data) = fs::read_to_string(&save_file) {
                    if let Ok(save_data) = serde_json::from_str::<ModelSave>(&json_data) {
                        best_all_time_network = save_data.network;
                        best_all_time_score = save_data.average_score;
                        best_all_time_max = save_data.max_score;
                    } else if let Ok(champion) = serde_json::from_str::<Network>(&json_data) {
                        best_all_time_network = champion;
                    }
                }
                let fallback_networks = vec![best_all_time_network.clone(); pop_size];
                population.next_generation(fallback_networks);
                continue;
            }
        }

        let sorted_networks: Vec<Network> = scored_agents.into_iter().map(|(_, _, net)| net).collect();
        population.next_generation(sorted_networks);
    }
}