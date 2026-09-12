use rand::Rng;

pub const MUT_CHANCE: f32 = 0.05;


struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

struct Layer {
    neurons : Vec<Neuron>,
}

struct Network {
    layers : Vec<Layer>,
}

impl Neuron {
    fn new(entry_size: usize, weights_given: Option<Vec<f32>>, bias_given: Option<f32>) -> Self {
        let mut weights = match weights_given {
            Some(w) => w,
            None => vec![],
        };
        
        while entry_size > weights.len() {
            let mut rng = rand::thread_rng();
            weights.push(rng.gen_range(-2.0..2.0));
        }
        while entry_size < weights.len() {
            let _ = weights.pop();
        }

        let bias = match bias_given {
            Some(value) => value, 
            None => {
                let mut rng = rand::thread_rng();
                rng.gen_range(-1.0..1.0)
            }
        };

        Self {
            weights,
            bias,
        }
    }

    fn logic(&self, entries: &Vec<f32>) -> f32 {
        let mut total: f32 = 0.0;
        for i in 0..entries.len() {
            total += entries[i] * self.weights[i];
        }
        total += self.bias;
        return total;
    }

}

impl Layer {
    fn new(size: usize, entry_size: usize, mut weights: Option<Vec<Vec<f32>>>, mut bias_given: Option<Vec<f32>>) -> Self {
        let mut neurons: Vec<Neuron> = vec![];
        for _ in 0..size {
            neurons.push(Neuron::new(entry_size, weights.as_mut().and_then(|w| w.pop()), bias_given.as_mut().and_then(|b| b.pop())));
        }
        Self {
            neurons
        }
    }

    fn logic(&self, entries: Vec<f32>) -> Vec<f32> {
        let mut results: Vec<f32> = vec![];
        for i in 0..self.neurons.len() {
            results.push(self.neurons[i].logic(entries));
        }
        return results;
    }
}

impl Network {
    fn new() -> Self {
        
    }
}