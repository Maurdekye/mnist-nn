#![feature(array_windows)]
#![feature(iterator_try_collect)]

use std::{
    error::Error,
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
};

use clap::{Parser, Subcommand};

mod nn;

use nn::*;

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, Box<dyn Error>> {
    let mut uint = [0; 4];
    reader.read(&mut uint)?;
    Ok(u32::from_be_bytes(uint))
}

fn load_mnist_images(path: PathBuf) -> Result<Vec<Vec<Precision>>, Box<dyn Error>> {
    let mut images = BufReader::new(File::open(path)?);
    if read_u32(&mut images)? != 0x00000803 {
        Err("Magic number does not match on images file")?;
    }
    let num_items = read_u32(&mut images)?;
    let height = read_u32(&mut images)?;
    let width = read_u32(&mut images)?;

    let mut image_samples = Vec::new();
    for _ in 0..num_items {
        let mut image = vec![0; (width * height) as usize];
        images.read(&mut image)?;
        let sample = image
            .into_iter()
            .map(|opacity| (opacity as Precision) / 255.0)
            .collect();
        image_samples.push(sample);
    }

    Ok(image_samples)
}

fn load_mnist_labels(path: PathBuf) -> Result<Vec<Vec<Precision>>, Box<dyn Error>> {
    let mut labels = BufReader::new(File::open(path)?);
    if read_u32(&mut labels)? != 0x00000801 {
        Err("Magic number does not match on labels file")?;
    }
    let _num_items = read_u32(&mut labels)?;

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

fn train_mnist(args: TrainMnistArgs) -> Result<(), Box<dyn Error>> {
    let mut model = match args.model {
        Some(path) => {
            println!("Loadinging model from {}", path.to_string_lossy());
            load(path)?
        }
        None => {
            println!("Initializing new model");
            Model::new(vec![784, 32, 10])
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
        args.temperature,
        Some(args.batch_size),
    );
    println!("Training finished");

    println!("Saving model");
    save(&model, args.save)?;

    println!("Done");

    Ok(())
}

fn train_xor(args: TrainXorArgs) -> Result<(), Box<dyn Error>> {
    let mut model = match args.model {
        Some(path) => {
            println!("Loadinging model from {}", path.to_string_lossy());
            load(path)?
        }
        None => {
            println!("Initializing new model");
            Model::new(vec![2, 2, 1])
        }
    };

    let samples = vec![
        Sample(vec![0.0, 0.0], vec![0.0]),
        Sample(vec![0.0, 1.0], vec![1.0]),
        Sample(vec![1.0, 0.0], vec![1.0]),
        Sample(vec![1.0, 1.0], vec![0.0]),
    ];

    println!("Beginning training");
    train::<SiLu>(&mut model, &samples, args.steps, args.temperature, None);
    println!("Training finished");

    println!("Saving model");
    save(&model, args.save)?;

    println!("Done");

    Ok(())
}

fn inference(args: InferenceArgs) -> Result<(), Box<dyn Error>> {
    let model = load(args.model)?;
    println!("{:?}", args.input);
    let output = model.inference::<SiLu>(&args.input);
    println!("{output:?}");
    Ok(())
}

#[derive(Parser)]
struct Args {
    #[clap(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    TrainMnist(TrainMnistArgs),
    TrainXor(TrainXorArgs),
    Inference(InferenceArgs),
}

#[derive(Parser)]
struct TrainMnistArgs {
    #[clap(short = 'd', long, default_value = "mnist")]
    mnist_data: PathBuf,

    #[clap(short, long)]
    model: Option<PathBuf>,

    #[clap(short, long)]
    save: PathBuf,

    #[clap(short = 'i', long, default_value_t = 1000)]
    steps: usize,

    #[clap(short, long, default_value_t = 1e-7)]
    temperature: Precision,

    #[clap(short, long, default_value_t = 100)]
    batch_size: usize,
}

#[derive(Parser)]
struct TrainXorArgs {
    #[clap(short, long)]
    model: Option<PathBuf>,

    #[clap(short, long)]
    save: PathBuf,

    #[clap(short = 'i', long, default_value_t = 1000)]
    steps: usize,

    #[clap(short, long, default_value_t = 1e-2)]
    temperature: Precision,
}

#[derive(Parser)]
struct InferenceArgs {
    model: PathBuf,

    input: Vec<Precision>,
}

fn main() -> Result<(), Box<dyn Error>> {
    let args = Args::parse();

    match args.command {
        Command::TrainMnist(args) => train_mnist(args),
        Command::TrainXor(args) => train_xor(args),
        Command::Inference(args) => inference(args),
    }
}
