mod ia;
mod manager;
mod game;

use game::CarDodgeEnv;
use manager::Population;
use ia::Network;
use std::fs;
use std::path::Path;
use rayon::prelude::*; // <-- L'import magique pour le multithreading

fn main() {
    println!("Démarrage de l'entraînement IA (Mode Multithread activé) !");

    let pop_size = 100;
    let per_layer = vec![6, 3]; 
    let num_games = 100;
    let save_file = "best_model.json";

    // 1. GESTION DE LA SAUVEGARDE (CHARGEMENT)
    let mut initial_agents = None;
    if Path::new(save_file).exists() {
        println!("Sauvegarde trouvée ! Chargement de {}...", save_file);
        if let Ok(json_data) = fs::read_to_string(save_file) {
            if let Ok(champion) = serde_json::from_str::<Network>(&json_data) {
                initial_agents = Some(vec![champion; pop_size]);
                println!("Modèle chargé avec succès !");
            }
        }
    } else {
        println!("Aucune sauvegarde trouvée. Création d'une nouvelle lignée.");
    }

    let mut population = Population::new(pop_size, per_layer, 8, initial_agents);
    let mut best_all_time_network = population.agents[0].clone();
    let mut best_all_time_score = 0.0;

    for _ in 0..1000 {
        // 2. LA BOUCLE PARALLÈLE
        // Au lieu d'une boucle "for", on demande à Rayon de lancer les parties en même temps
        let total_scores = (0..num_games)
            .into_par_iter() // <-- C'est ici que ton PC passe à 100% d'utilisation
            .map(|_| {
                let mut env = CarDodgeEnv::new(pop_size);

                while !env.done {
                    let states = env.get_states();
                    let mut actions: Vec<u8> = vec![0; pop_size];

                    for i in 0..pop_size {
                        if let Some(state_array) = states[i] {
                            let entries = state_array.to_vec();
                            // Les cerveaux sont lus simultanément par tous les cœurs
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

                // À la fin d'UNE partie, ce cœur renvoie les scores des 100 voitures
                env.cars.iter().map(|car| car.time_alive).collect::<Vec<f32>>()
            })
            // Récupère les tableaux de scores de toutes les parties et les additionne
            .reduce(
                || vec![0.0; pop_size],
                |mut acc, scores| {
                    for i in 0..pop_size {
                        acc[i] += scores[i];
                    }
                    acc
                },
            );

        // 3. MOYENNE ET ÉVOLUTION
        let mut scored_agents: Vec<(f32, Network)> = vec![];
        for i in 0..pop_size {
            let average_score = total_scores[i] / (num_games as f32);
            scored_agents.push((average_score, population.agents[i].clone()));
        }

        scored_agents.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        println!(
            "Génération {} - Meilleur temps MOYEN : {:.2}s", 
            population.generation, 
            scored_agents[0].0
        );

        if scored_agents[0].0 > best_all_time_score {
            best_all_time_score = scored_agents[0].0;
            best_all_time_network = scored_agents[0].1.clone();
        }

        let sorted_networks: Vec<Network> = scored_agents.into_iter().map(|(_, net)| net).collect();
        population.next_generation(sorted_networks);
    }
    
    // 4. GESTION DE LA SAUVEGARDE (ÉCRITURE)
    println!("Entraînement terminé ! Sauvegarde du meilleur modèle...");
    if let Ok(json_data) = serde_json::to_string_pretty(&best_all_time_network) {
        if fs::write(save_file, json_data).is_ok() {
            println!("Modèle sauvegardé avec succès dans '{}' (Record : {:.2}s) !", save_file, best_all_time_score);
        } else {
            println!("Erreur : Impossible d'écrire sur le disque.");
        }
    }
}