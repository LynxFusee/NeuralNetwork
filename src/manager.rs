use crate::ia::Network;

pub struct Population {
    pub agents: Vec<Network>,
    pub generation: u32,
}

impl Population {
    pub fn new(population_size: usize, per_layer: Vec<usize>, entry_size: usize, saved_agents: Option<Vec<Network>>) -> Self {
        if let Some(agents) = saved_agents {
            return Self {
                agents,
                generation: 0,
            };
        }

        let mut agents: Vec<Network> = vec![];
        for _ in 0..population_size {
            agents.push(Network::new(per_layer.clone(), entry_size, None, None));
        }
        
        Self {
            agents,
            generation: 0,
        }
    }

    pub fn next_generation(&mut self, sorted_agents: Vec<Network>) {
        let population_size = sorted_agents.len();
        let mut next_gen: Vec<Network> = vec![];
        let champion = sorted_agents[0].clone();
        next_gen.push(champion);

        let mut top_performers_count = (population_size as f32 * 0.20) as usize;
        if top_performers_count < 1 {
            top_performers_count = 1;
        }

        while next_gen.len() < population_size {
            for i in 0..top_performers_count {
                if next_gen.len() >= population_size {
                    break;
                }
                
                let mut child = sorted_agents[i].clone();
                child.mutate(); 
                next_gen.push(child);
            }
        }

        self.agents = next_gen;
        self.generation += 1;
    }
}