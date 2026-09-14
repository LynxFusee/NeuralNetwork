use rand::RngExt;
use rand_distr::{Normal, Distribution};
use serde::{Serialize, Deserialize};

pub const MUT_CHANCE: f32 = 0.50; 
pub const MUT_FORCE: f32 = 0.30;  
//pub const TOP_MUT_CHANCE: f32 = 0.03;
pub const RND_MUT_CHANCE: f32 = 0.85;

#[derive(Clone, Serialize, Deserialize)]
pub struct Neuron {
    pub weights: Vec<f32>,
    pub bias: f32,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Layer {
    pub neurons: Vec<Neuron>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Network {
    pub layers: Vec<Layer>,
}

impl Neuron {
    pub fn new(entry_size: usize, weights_given: Option<Vec<f32>>, bias_given: Option<f32>) -> Self {
        let mut rng = rand::rng();
        let mut weights = match weights_given {
            Some(w) => w,
            None => vec![],
        };

        while entry_size > weights.len() {
            weights.push(rng.random_range(-2.0..2.0));
        }

        while entry_size < weights.len() {
            let _ = weights.pop();
        }

        let bias = match bias_given {
            Some(value) => value, 
            None => {
                rng.random_range(-1.0..1.0)
            }
        };

        Self {
            weights,
            bias,
        }
    }

    pub fn logic(&self, entries: &Vec<f32>) -> f32 {
        let mut total: f32 = entries.iter().zip(self.weights.iter()).map(|(e, w)| e * w).sum();
        total += self.bias;
        total.tanh()
    }

    pub fn mutate(&mut self) {
        let mut rng = rand::rng();
        let dist = Normal::new(0.0, MUT_FORCE).unwrap(); 
        for weight in self.weights.iter_mut() {
            if rng.random::<f32>() < MUT_CHANCE {
                if rng.random::<f32>() < RND_MUT_CHANCE {
                    *weight += dist.sample(&mut rng);
                    *weight = weight.clamp(-2.0, 2.0);
                } else {
                    *weight = rng.random_range(-2.0..2.0);
                }
            }
        }
        if rng.random::<f32>() < MUT_CHANCE {
            if rng.random::<f32>() < RND_MUT_CHANCE {
                self.bias += dist.sample(&mut rng);
                self.bias = self.bias.clamp(-2.0, 2.0);
            } else {
                self.bias = rng.random_range(-2.0..2.0);
            }
        }
    }
}

impl Layer {
    pub fn new(size: usize, entry_size: usize, mut weights: Option<Vec<Vec<f32>>>, mut bias_given: Option<Vec<f32>>) -> Self {
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

    pub fn logic(&self, entries: Vec<f32>) -> Vec<f32> {
        self.neurons.iter().map(|neuron| neuron.logic(&entries)).collect()
    }

    pub fn mutate(&mut self) {
        for neuron in self.neurons.iter_mut() {
            neuron.mutate();
        }
    }
}

impl Network {
    pub fn new(per_layer: Vec<usize>, mut entry_size: usize, mut weights: Option<Vec<Vec<Vec<f32>>>>, mut bias_given: Option<Vec<Vec<f32>>>) -> Self {
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

    pub fn logic(&self, mut entries: Vec<f32>) -> Vec<f32> {
        for i in 0..self.layers.len() {
            entries = self.layers[i].logic(entries);
        }
        entries
    }

    pub fn mutate(&mut self) {
        for layer in self.layers.iter_mut() {
            layer.mutate();
        }
    }
}