#![feature(array_windows)]
#![feature(iterator_try_collect)]

use std::{
    error::Error,
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

use clap::Parser;

mod nn;

use nn::*;

fn load_mnist_images(path: PathBuf) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut images = BufReader::new(File::open(path)?);
    let mut magic_number = [0; 4];
    images.read(&mut magic_number)?;
    if u32::from_be_bytes(magic_number) != 0x00000803 {
        Err("Magic number does not match on images file")?;
    }
    let mut num_items = [0; 4];
    images.read(&mut num_items)?;
    let num_items = u32::from_be_bytes(num_items);
    let mut height = [0; 4];
    images.read(&mut height)?;
    let height = u32::from_be_bytes(height);
    let mut width = [0; 4];
    images.read(&mut width)?;
    let width = u32::from_be_bytes(width);

    let mut image_samples = Vec::new();
    for _ in 0..num_items {
        let mut image = vec![0; (width * height) as usize];
        images.read(&mut image)?;
        let sample = image
            .into_iter()
            .map(|opacity| (opacity as f32) / 255.0)
            .collect();
        image_samples.push(sample);
    }

    Ok(image_samples)
}

fn load_mnist_labels(path: PathBuf) -> Result<Vec<Vec<f32>>, Box<dyn Error>> {
    let mut labels = BufReader::new(File::open(path)?);
    let mut magic_number = [0; 4];
    labels.read(&mut magic_number)?;
    if u32::from_be_bytes(magic_number) != 0x00000801 {
        Err("Magic number does not match on labels file")?;
    }
    let mut num_items = [0; 4];
    labels.read(&mut num_items)?;
    let _num_items = u32::from_be_bytes(num_items);

    labels
        .bytes()
        .map(|label| {
            let label = label?;
            let mut label_vector = vec![0.0; 10];
            label_vector[label as usize] = 1.0;
            Ok(label_vector)
        })
        .try_collect()
}

fn load_mnist(images: PathBuf, labels: PathBuf) -> Result<Vec<Sample>, Box<dyn Error>> {
    let images = load_mnist_images(images)?;
    let labels = load_mnist_labels(labels)?;
    Ok(images
        .into_iter()
        .zip(labels)
        .map(|(i, l)| Sample(i, l))
        .collect())
}

#[derive(Parser)]
struct Args {
    #[clap(short = 'd', long, default_value = "mnist")]
    mnist_data: PathBuf,

    #[clap(short, long)]
    model: Option<PathBuf>,

    #[clap(short, long)]
    save: PathBuf,

    #[clap(short = 'i', long, default_value_t = 10_000)]
    steps: usize,

    #[clap(short, long, default_value_t = 1e-6)]
    temperature: Precision,

    #[clap(short, long, default_value_t = 1e-6)]
    epsilon: Precision,

    #[clap(short, long, default_value_t = 100)]
    batch_size: usize,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    let mut model = match args.model {
        Some(path) => {
            println!("Loadinging model from {}", path.to_string_lossy());
            load(path)?
        }
        None => {
            println!("Initializing new model");
            Model::new(vec![784, 256, 10])
        }
    };

    println!("Loading training data");
    let samples = load_mnist(
        args.mnist_data.join("train-images.idx3-ubyte"),
        args.mnist_data.join("train-labels.idx1-ubyte"),
    )?;

    println!("Beginning training");
    train::<SiLu>(
        &mut model,
        &samples,
        args.steps,
        args.epsilon,
        args.temperature,
        args.batch_size,
    );
    println!("Training finished");

    println!("Saving model");
    save(&model, args.save)?;

    println!("Done");

    Ok(())
}
