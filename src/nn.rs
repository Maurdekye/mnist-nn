use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter},
    iter::zip,
    ops::Range,
    path::PathBuf,
};

use progress_observer::reprint;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
pub type Precision = f64;

#[derive(Debug, Serialize, Deserialize)]
pub struct Model {
    arch: Vec<usize>,
    weights: Vec<Vec<Precision>>,
}

impl Model {
    pub fn new(arch: Vec<usize>) -> Self {
        let mut rng = StdRng::seed_from_u64(0);

        let mut weights = Vec::new();

        for &[x, y] in arch.array_windows() {
            weights.push((0..((x + 1) * y)).map(|_| rng.gen()).collect());
        }

        Self { arch, weights }
    }

    pub fn forward<A>(
        &self,
        features: &Vec<Precision>,
    ) -> (Vec<Precision>, Vec<Vec<Precision>>, Vec<Vec<Precision>>)
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

        let mut network_inputs = Vec::new();
        let mut network_outputs = Vec::new();
        let mut activations = features.clone();
        for (weights, &[from, _to]) in self.weights.iter().zip(self.arch.array_windows()) {
            activations.push(1.0);
            network_outputs.push(activations.clone());
            activations = weights
                .chunks(from + 1)
                .map(|weights| {
                    weights
                        .iter()
                        .zip(&*activations)
                        .map(|(weight, activation)| weight * activation)
                        .sum()
                })
                .collect();
            network_inputs.push(activations.clone());
            activations = activations.into_iter().map(A::activate).collect();
        }

        (activations, network_inputs, network_outputs)
    }

    pub fn inference<A: ActivationFunction>(&self, features: &Vec<Precision>) -> Vec<Precision> {
        self.forward::<A>(features).0
    }

    pub fn _backward<A: ActivationFunction>(
        &self,
        label: &Vec<Precision>,
        output: &Vec<Precision>,
        network_inputs: &Vec<Vec<Precision>>,
        network_outputs: &Vec<Vec<Precision>>,
    ) -> Vec<Vec<Precision>> {
        let mut gradients = Vec::new();
        let mut delta: Vec<Precision> = Vec::new();
        for (i, rev_i) in (0..self.weights.len()).zip((0..self.weights.len()).rev()) {
            let d_activ: Vec<Precision> = network_inputs[rev_i]
                .iter()
                .map(|input| A::derivative(*input))
                .collect();
            let d_error: Vec<Precision> = if i == 0 {
                label
                    .iter()
                    .zip(output)
                    .map(|(label, output)| output - label)
                    .collect()
            } else {
                (&self.weights[rev_i + 1])
                    .transposed_chunks((&self.weights[rev_i + 1]).len() / delta.len())
                    .take(d_activ.len())
                    .map(|weights| {
                        weights
                            .iter()
                            .zip(&delta)
                            .map(|(&weight, delta)| weight * delta)
                            .sum::<Precision>()
                    })
                    .collect()
            };
            delta = d_activ
                .into_iter()
                .zip(d_error)
                .map(|(activation, error)| activation * error)
                .collect();
            let gradient: Vec<Precision> = delta
                .iter()
                .flat_map(|delta| {
                    network_outputs[rev_i]
                        .iter()
                        .map(move |&output| delta * output)
                })
                .collect();
            gradients.push(gradient);
        }

        gradients.into_iter().rev().collect()
    }

    pub fn backward<A: ActivationFunction>(
        &self,
        label: &Vec<Precision>,
        output: &Vec<Precision>,
        network_inputs: &Vec<Vec<Precision>>,
        network_outputs: &Vec<Vec<Precision>>,
    ) -> Vec<Vec<Precision>> {
        let mut delta: Vec<Precision> = network_inputs
            .last()
            .unwrap()
            .iter()
            .zip(label)
            .zip(output)
            .map(|((&input, &label), &output)| A::derivative(input) * (output - label))
            .collect();
        let last_output = network_outputs.last().unwrap();
        let mut gradients: Vec<Vec<Precision>> = vec![delta
            .iter()
            .flat_map(|output| last_output.iter().map(|delta| delta * *output))
            .collect()];
        for (((inputs, outputs), weights), &width) in network_inputs
            .iter()
            .rev()
            .skip(1)
            .zip(network_outputs.iter().rev().skip(1))
            .zip(self.weights.iter().rev())
            .zip(self.arch.iter().rev().skip(1))
        {
            delta = inputs
                .iter()
                .zip(
                    weights
                        .transposed_chunks(weights.len() / delta.len())
                        .take(width)
                        .map(|weights| {
                            weights
                                .iter()
                                .zip(&delta)
                                .map(|(&weight, delta)| weight * delta)
                                .sum::<Precision>()
                        }),
                )
                .map(|(&input, error)| A::derivative(input) * error)
                .collect();
            let gradient: Vec<Precision> = delta
                .iter()
                .flat_map(|delta| outputs.iter().map(move |&output| delta * output))
                .collect();
            gradients.push(gradient);
        }

        gradients.into_iter().rev().collect()
    }

    pub fn forward_backward<A: ActivationFunction>(
        &self,
        Sample(features, label): &Sample,
    ) -> Gradient {
        let (output, network_inputs, network_outputs) = self.forward::<A>(features);
        let gradients = self.backward::<A>(label, &output, &network_inputs, &network_outputs);
        Gradient(gradients)
    }

    fn print_params(&self) {
        for i in 0..self.arch.len() {
            println!();
            let subtitle = if i == 0 {
                " (input)"
            } else if i == self.arch.len() - 1 {
                " (output)"
            } else {
                ""
            };
            println!("Layer {i}{subtitle}:");
            println!("Width: {}", self.arch[i]);
            if i >= 1 {
                self.print_weights(i - 1);
            }
        }
    }

    fn print_weights(&self, index: usize) {
        println!("Weights:");
        for chunk in self.weights[index].chunks(self.arch[index] + 1) {
            println!(
                "{}",
                chunk[..chunk.len() - 1]
                    .iter()
                    .map(|weight| format!("{weight: <6.3}"))
                    .collect::<Vec<_>>()
                    .join("")
            );
        }
        println!("Biases:");
        println!(
            "{}",
            self.weights[index]
                .chunks(self.arch[index] + 1)
                .map(|chunk| chunk[chunk.len() - 1])
                .map(|bias| format!("{bias: <6.3}"))
                .collect::<Vec<_>>()
                .join("")
        );
    }
}

#[derive(Debug)]
pub struct Gradient(Vec<Vec<Precision>>);

pub struct Sample(pub Vec<Precision>, pub Vec<Precision>);

pub fn loss<A: ActivationFunction>(model: &Model, Sample(input, label): &Sample) -> Precision {
    label
        .iter()
        .zip(model.inference::<A>(input))
        .map(|(&label, actual)| (label - actual).powi(2))
        .sum::<Precision>()
        / (label.len() * 2) as Precision
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
    let mut gradient = Gradient(Vec::new());
    let base_error = batch_error::<A>(model, samples);
    for layer in 0..model.weights.len() {
        let mut layer_weight_gradients = vec![0.0; model.weights[layer].len()];
        for weight in 0..layer_weight_gradients.len() {
            reprint!(
                "layer {layer} {:.2}%",
                (weight as f32 / layer_weight_gradients.len() as f32) * 100.0
            );
            let original_weight = model.weights[layer][weight];
            model.weights[layer][weight] += epsilon;
            let nudged_error = batch_error::<A>(model, samples);
            model.weights[layer][weight] = original_weight;
            layer_weight_gradients[weight] = (nudged_error - base_error) / epsilon;
        }
        gradient.0.push(layer_weight_gradients);
    }
    reprint!("");
    (base_error, gradient)
}

pub fn apply_gradient<A: ActivationFunction>(
    model: &mut Model,
    gradient: Gradient,
    temperature: Precision,
) {
    for (model_weights, gradient_weights) in model.weights.iter_mut().zip(gradient.0) {
        for (model_weight, gradient_weight) in model_weights.iter_mut().zip(gradient_weights) {
            *model_weight -= gradient_weight * temperature;
        }
    }
}

pub fn train<A: ActivationFunction>(
    model: &mut Model,
    samples: &Vec<Sample>,
    steps: usize,
    epsilon: Precision,
    temperature: Precision,
    batch_size: Option<usize>,
) {
    for i in 0..steps {
        let batch: Vec<&Sample> = match batch_size {
            Some(batch_size) => samples
                .choose_multiple(&mut thread_rng(), batch_size)
                .collect(),
            None => samples.iter().collect(),
        };
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

#[derive(Serialize, Deserialize)]
pub struct ReLu;
impl ActivationFunction for ReLu {
    fn activate(x: Precision) -> Precision {
        x.max(0.0)
    }

    fn derivative(x: Precision) -> Precision {
        if x > 0.0 {
            1.0
        } else {
            0.0
        }
    }
}

#[derive(Serialize, Deserialize)]
pub struct LeakyReLu;
impl ActivationFunction for LeakyReLu {
    fn activate(x: Precision) -> Precision {
        x.max(0.01 * x)
    }

    fn derivative(x: Precision) -> Precision {
        if x > 0.0 {
            1.0
        } else {
            0.01
        }
    }
}

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

pub struct TransposedChunks<'a, T> {
    slice: &'a [T],
    step: usize,
    progress: Range<usize>,
}

impl<'a, T> Iterator for TransposedChunks<'a, T> {
    type Item = Vec<&'a T>;

    fn next(&mut self) -> Option<Self::Item> {
        self.progress
            .next()
            .map(|offset| self.slice.iter().skip(offset).step_by(self.step).collect())
    }
}

pub trait AsTransposeChunks: Sized {
    type Item;

    fn transposed_chunks(&self, num_chunks: usize) -> TransposedChunks<Self::Item>;
}

impl<T> AsTransposeChunks for &[T] {
    type Item = T;

    fn transposed_chunks(&self, num_chunks: usize) -> TransposedChunks<Self::Item> {
        let step = num_chunks;
        TransposedChunks {
            slice: self,
            step,
            progress: 0..step,
        }
    }
}

impl<T> AsTransposeChunks for &Vec<T> {
    type Item = T;

    fn transposed_chunks(&self, num_chunks: usize) -> TransposedChunks<Self::Item> {
        let step = num_chunks;
        TransposedChunks {
            slice: self,
            step,
            progress: 0..step,
        }
    }
}

#[test]
fn transposed_chunks() {
    let a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12];
    println!("chunks:");
    for chunk in a.chunks(3) {
        println!("{chunk:?}");
    }
    println!("transposed chunks:");
    for chunk in (&a[..]).transposed_chunks(2) {
        println!("{chunk:?}");
    }
}

#[test]
fn backprop() {
    let mut model = Model::new(vec![2, 2, 1]);
    model.print_params();

    let test_sample = Sample(vec![0.0, 1.0], vec![1.0]);

    let (_, naive_gradient) = compute_gradient::<SiLu>(&mut model, &vec![&test_sample], 1e-12);
    let backprop_gradient = model.forward_backward::<SiLu>(&test_sample);
    dbg!(naive_gradient);
    dbg!(backprop_gradient);
}
