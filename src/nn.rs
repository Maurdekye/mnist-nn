use std::{
    error::Error,
    fs::File,
    io::{BufReader, BufWriter},
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
        let mut rng = thread_rng();

        let mut weights = Vec::new();

        for &[x, y] in arch.array_windows() {
            weights.push((0..((x + 1) * y)).map(|_| rng.gen()).collect());
        }

        Self { arch, weights }
    }

    pub fn forward<A>(&self, features: &Vec<Precision>) -> Vec<Vec<Precision>>
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

        let mut activations = vec![features.clone()];
        for (weights, &[from, _to]) in self.weights.iter().zip(self.arch.array_windows()) {
            let boundary = activations.last_mut().unwrap();
            boundary.push(1.0);
            dbg!(weights.chunks(from + 1).collect::<Vec<_>>());
            let next_boundary = weights
                .chunks(from + 1)
                .map(|weights| {
                    A::activate(
                        weights
                            .iter()
                            .zip(&*boundary)
                            .map(|(weight, activation)| weight * activation)
                            .sum(),
                    )
                })
                .collect();
            activations.push(next_boundary);
        }

        activations
    }

    pub fn inference<A: ActivationFunction>(&self, features: &Vec<Precision>) -> Vec<Precision> {
        self.forward::<A>(features).pop().unwrap()
    }

    pub fn backward<A: ActivationFunction>(
        &self,
        label: &Vec<Precision>,
        mut activations: Vec<Vec<Precision>>,
    ) -> Gradient {
        let last_activation = activations.pop().unwrap();
        let last_delta: Vec<_> = last_activation
            .iter()
            .zip(label)
            .map(|(activation, label)| (activation - label) * A::derivative(*activation))
            .collect();
        dbg!(&last_delta);
        let mut deltas = vec![last_delta];
        dbg!(&activations);

        for (activations, weights) in activations.into_iter().zip(&self.weights).rev() {
            dbg!(&activations);
            dbg!(weights);
            let delta = deltas.last().unwrap();
            dbg!(delta);
            dbg!(weights
                .transposed_chunks(weights.len() / deltas.len())
                .collect::<Vec<_>>());
            debug_assert_eq!(
                weights
                    .transposed_chunks(weights.len() / deltas.len())
                    .count(),
                activations.len()
            );

            let new_delta: Vec<Precision> = weights
                .transposed_chunks(weights.len() / deltas.len())
                .zip(activations)
                .map(|(weights, activation)| {
                    dbg!(&weights);
                    dbg!(activation);
                    // debug_assert_eq!(weights.len(), delta.len());
                    let delta: Precision = weights
                        .iter()
                        .zip(delta)
                        .map(|(&weight, delta)| weight * delta)
                        .sum();
                    dbg!(&delta);
                    delta * A::derivative(activation)
                })
                .collect();
            dbg!(&new_delta);
            deltas.push(new_delta);
        }

        deltas.reverse();

        Gradient(deltas)
    }

    pub fn forward_backward<A: ActivationFunction>(
        &self,
        Sample(features, label): &Sample,
    ) -> Gradient {
        let activations = self.forward::<A>(features);
        self.backward::<A>(label, activations)
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
    let mut model = Model {
        arch: vec![
            2,
            2,
            1,
        ],
        weights: vec![
            vec![
                0.4988102959636601,
                0.9047553176084865,
                0.5845558959510986,
                0.3004672416053119,
                0.6780994008327456,
                0.42221830855443954,
            ],
            vec![
                0.8067167072599537,
                0.5558960863086262,
                0.8891306385486438,
            ],
        ],
    };
    dbg!(&model);

    let test_sample = Sample(vec![0.0, 1.0], vec![1.0]);

    let (_, naive_gradient) = compute_gradient::<SiLu>(&mut model, &vec![&test_sample], 1e-12);
    dbg!(naive_gradient);
    let backprop_gradient = model.forward_backward::<SiLu>(&test_sample);
    dbg!(backprop_gradient);
}

