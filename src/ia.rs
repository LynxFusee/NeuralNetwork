use rand::Rng;
use rand_distr::{Normal, Distribution};

pub const MUT_CHANCE: f32 = 0.20;
pub const MUT_FORCE: f32 = 0.05;
pub const TOP_MUT_CHANCE: f32 = 0.03;

struct Neuron {
    weights: Vec<f32>,
    bias: f32,
}

struct Layer {
    neurons: Vec<Neuron>,
}

struct Network {
    layers: Vec<Layer>,
}

impl Neuron {
    fn new(entry_size: usize, weights_given: Option<Vec<f32>>, bias_given: Option<f32>) -> Self {
        let mut rng = rand::thread_rng();
        let mut weights = match weights_given {
            Some(w) => w,
            None => vec![],
        };

        while entry_size > weights.len() {
            weights.push(rng.gen_range(-2.0..2.0));
        }

        while entry_size < weights.len() {
            let _ = weights.pop();
        }

        let bias = match bias_given {
            Some(value) => value, 
            None => {
                rng.gen_range(-1.0..1.0)
            }
        };

        Self {
            weights,
            bias,
        }
    }

    fn logic(&self, entries: &Vec<f32>) -> f32 {
        let mut total: f32 = entries.iter().zip(self.weights.iter()).map(|(e, w)| e * w).sum();
        total += self.bias;
        total.tanh()
    }

    fn mutate(&mut self) {
        let mut rng = rand::thread_rng();
        let dist = Normal::new(0.0, MUT_FORCE).unwrap(); 
        for weight in self.weights.iter_mut() {
            if rng.gen::<f32>() < MUT_CHANCE {
                *weight += dist.sample(&mut rng);
                *weight = weight.clamp(-2.0, 2.0);
            }
        }
        if rng.gen::<f32>() < MUT_CHANCE {
            self.bias += dist.sample(&mut rng);
            self.bias = self.bias.clamp(-2.0, 2.0);
        }
    }
}

impl Layer {
    fn new(size: usize, entry_size: usize, mut weights: Option<Vec<Vec<f32>>>, mut bias_given: Option<Vec<f32>>) -> Self {
        if let Some(w) = &mut weights {
            w.reverse();
        }
        if let Some(b) = &mut bias_given {
            b.reverse();
        }
        let mut neurons: Vec<Neuron> = vec![];
        for _ in 0..size {
            neurons.push(Neuron::new(
                entry_size, 
                weights.as_mut().and_then(|w| w.pop()), 
                bias_given.as_mut().and_then(|b| b.pop())
            ));
        }
        Self {
            neurons
        }
    }

    fn logic(&self, entries: Vec<f32>) -> Vec<f32> {
        self.neurons.iter().map(|neuron| neuron.logic(&entries)).collect()
    }

    fn mutate(&mut self) {
        for i in 0..self.neurons.len() {
            self.neurons[i].mutate();
        }
    }

    fn upgrade(&mut self) {
        
    }
}

impl Network {
    fn new(per_layer: Vec<usize>, mut entry_size: usize, mut weights: Option<Vec<Vec<Vec<f32>>>>, mut bias_given: Option<Vec<Vec<f32>>>) -> Self {
        if let Some(w) = &mut weights {
            w.reverse();
        }
        if let Some(b) = &mut bias_given {
            b.reverse();
        }
        let mut layers: Vec<Layer> = vec![];
        for i in 0..per_layer.len() {
            layers.push(Layer::new(per_layer[i], entry_size, weights.as_mut().and_then(|w| w.pop()), bias_given.as_mut().and_then(|b| b.pop())));
            entry_size = per_layer[i];
        }
        Self {
            layers
        }
    }

    fn logic(&self, mut entries: Vec<f32>) -> Vec<f32> {
        for i in 0..self.layers.len() {
            entries = self.layers[i].logic(entries);
        }
        entries
    }
}