use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter},
    path::PathBuf,
};

use progress_observer::reprint;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
pub type Precision = f64;

#[derive(Serialize, Deserialize)]
pub struct Model {
    arch: Vec<usize>,
    weights: Vec<Vec<Precision>>,
    biases: Vec<Vec<Precision>>,
}

impl Model {
    pub fn new(arch: Vec<usize>) -> Self {
        let mut rng = thread_rng();

        let mut weights = Vec::new();
        let mut biases = Vec::new();

        for &[x, y] in arch.array_windows() {
            weights.push((0..(x * y)).map(|_| rng.gen()).collect());
            biases.push((0..y).map(|_| rng.gen()).collect());
        }

        Self {
            arch,
            weights,
            biases,
        }
    }

    pub fn forward<A>(&self, features: Vec<Precision>) -> Vec<Precision>
    where
        A: ActivationFunction,
    {
        #[cfg(debug_assertions)]
        if features.len() != self.arch[0] {
            panic!(
                "input feature vector is the wrong size: expected {}, got {}",
                self.arch[0],
                features.len()
            );
        }

        let mut activations = features;
        for ((weights, biases), &[_from, to]) in self
            .weights
            .iter()
            .zip(&self.biases)
            .zip(self.arch.array_windows())
        {
            activations = activations
                .into_iter()
                .zip(weights.chunks(to))
                .zip(biases)
                .map(|((activation, weights), bias)| {
                    A::activate(
                        weights
                            .iter()
                            .map(|weight| weight * activation)
                            .sum::<Precision>()
                            + bias,
                    )
                })
                .collect();
        }
        activations
    }
}

pub struct Gradient {
    weights: Vec<Vec<Precision>>,
    biases: Vec<Vec<Precision>>,
}

pub struct Sample(pub Vec<Precision>, pub Vec<Precision>);

pub fn loss<A: ActivationFunction>(model: &Model, Sample(input, label): &Sample) -> Precision {
    label
        .iter()
        .zip(model.forward::<A>(input.clone()))
        .map(|(&label, actual)| (label - actual).powi(2))
        .sum::<Precision>()
        / label.len() as Precision
}

pub fn batch_error<A: ActivationFunction>(model: &Model, samples: &Vec<&Sample>) -> Precision {
    samples
        .iter()
        .map(|sample| loss::<A>(model, sample))
        .sum::<Precision>()
        / samples.len() as Precision
}

pub fn compute_gradient<A: ActivationFunction>(
    model: &mut Model,
    samples: &Vec<&Sample>,
    epsilon: Precision,
) -> (Precision, Gradient) {
    let mut gradient = Gradient {
        weights: Vec::new(),
        biases: Vec::new(),
    };
    let base_error = batch_error::<A>(model, samples);
    for layer in 0..model.weights.len() {
        // gradient for weights
        let mut layer_weight_gradients = vec![0.0; model.weights[layer].len()];
        for weight in 0..layer_weight_gradients.len() {
            reprint!(
                "layer {layer} weights {:.2}%",
                (weight as f32 / layer_weight_gradients.len() as f32) * 100.0
            );
            let original_weight = model.weights[layer][weight];
            model.weights[layer][weight] += epsilon;
            let nudged_error = batch_error::<A>(model, samples);
            model.weights[layer][weight] = original_weight;
            layer_weight_gradients[weight] = (nudged_error - base_error) / epsilon;
        }
        gradient.weights.push(layer_weight_gradients);

        // gradient for biases
        let mut layer_bias_gradients = vec![0.0; model.biases[layer].len()];
        for bias in 0..layer_bias_gradients.len() {
            reprint!(
                "layer {layer} biases {:.2}%",
                (bias as f32 / layer_bias_gradients.len() as f32) * 100.0
            );
            let original_bias = model.biases[layer][bias];
            model.biases[layer][bias] += epsilon;
            let nudged_error = batch_error::<A>(model, samples);
            model.biases[layer][bias] = original_bias;
            layer_bias_gradients[bias] = (nudged_error - base_error) / epsilon;
        }
        gradient.biases.push(layer_bias_gradients);
    }
    reprint!("");
    (base_error, gradient)
}

pub fn apply_gradient<A: ActivationFunction>(
    model: &mut Model,
    gradient: Gradient,
    temperature: Precision,
) {
    for (model_weights, gradient_weights) in model.weights.iter_mut().zip(gradient.weights) {
        for (model_weight, gradient_weight) in model_weights.iter_mut().zip(gradient_weights) {
            *model_weight -= gradient_weight * temperature;
        }
    }

    for (model_biases, gradient_biases) in model.biases.iter_mut().zip(gradient.biases) {
        for (model_bias, gradient_bias) in model_biases.iter_mut().zip(gradient_biases) {
            *model_bias -= gradient_bias * temperature;
        }
    }
}

pub fn train<A: ActivationFunction>(
    model: &mut Model,
    samples: &Vec<Sample>,
    steps: usize,
    epsilon: Precision,
    temperature: Precision,
    batch_size: usize,
) {
    for i in 0..steps {
        let batch: Vec<&Sample> = samples
            .choose_multiple(&mut thread_rng(), batch_size)
            .collect();
        let (loss, gradient) = compute_gradient::<A>(model, &batch, epsilon);
        apply_gradient::<A>(model, gradient, temperature);
        println!("step {i}: loss {loss}");
    }
}

pub fn save(model: &Model, filename: PathBuf) -> Result<(), Box<dyn Error>> {
    let file = File::create(filename)?;
    let file = BufWriter::new(file);
    bincode::serialize_into(file, model)?;
    Ok(())
}

pub fn load(filename: PathBuf) -> Result<Model, Box<dyn Error>> {
    let file = File::open(filename)?;
    let file = BufReader::new(file);
    let model = bincode::deserialize_from(file)?;
    Ok(model)
}

pub trait ActivationFunction {
    fn activate(x: Precision) -> Precision;
    fn derivative(x: Precision) -> Precision;
}

// #[derive(Serialize, Deserialize)]
// pub struct ReLu;
// impl ActivationFunction for ReLu {
//     fn activate(x: Precision) -> Precision {
//         x.max(0.0)
//     }
// }

// #[derive(Serialize, Deserialize)]
// pub struct LeakyReLu;
// impl ActivationFunction for LeakyReLu {
//     fn activate(x: Precision) -> Precision {
//         x.max(0.01 * x)
//     }
// }

#[derive(Serialize, Deserialize)]
pub struct SiLu;
impl ActivationFunction for SiLu {
    fn activate(x: Precision) -> Precision {
        x / (1.0 + (-x).exp())
    }

    fn derivative(x: Precision) -> Precision {
        let x_exp = x.exp();
        (x - 1.0) / (x_exp + 1.0) - (x / (x_exp + 1.0).powi(2)) + 1.0
    }
}
