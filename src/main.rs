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
use progress_observer::reprint;

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
            println!("Loading model from {}", path.to_string_lossy());
            let model = load(path)?;
            println!("Loaded model with architecture {:?}", model.arch());
            model
        }
        None => {
            let arch = args.arch.unwrap_or_else(|| vec![784, 256, 10]);
            println!("Initializing new model with architecture {arch:?}");
            Model::new(arch)
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
        args.batch_size,
        |model, i| {
            if let Some(save_every) = args.save_every {
                if i > 0 && i % save_every == 0 {
                    println!("Saving checkpoint");
                    let mut filename = args.save.clone();
                    filename.set_file_name(format!(
                        "{i}-{}",
                        filename.file_name().unwrap().to_string_lossy()
                    ));
                    match save(&model, filename) {
                        Ok(()) => println!("Checkpoint saved"),
                        Err(err) => println!("Error saving checkpoint: {err}"),
                    }
                }
            }
        },
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
            println!("Loading model from {}", path.to_string_lossy());
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
    train::<SiLu>(
        &mut model,
        &samples,
        args.steps,
        args.temperature,
        None,
        |_, _| {},
    );
    println!("Training finished");

    println!("Saving model");
    save(&model, args.save)?;

    println!("Done");

    Ok(())
}

fn test_mnist(args: TestMnistArgs) -> Result<(), Box<dyn Error>> {
    let model = load(args.model)?;

    println!("Loading testing data");
    let samples = load_mnist(
        args.mnist_data.join("t10k-images.idx3-ubyte"),
        args.mnist_data.join("t10k-labels.idx1-ubyte"),
    )?;
    let num_samples = samples.len();

    fn classify_label(output: &Vec<Precision>) -> usize {
        output
            .iter()
            .enumerate()
            .max_by(|(_, a), (_, b)| a.total_cmp(b))
            .map(|(i, _)| i)
            .unwrap_or(0)
    }

    let mut hits = 0;
    for (i, Sample(features, label)) in samples.iter().enumerate() {
        let prediction = model.inference::<SiLu>(features);
        let label_class = classify_label(label);
        let prediction_class = classify_label(&prediction);
        if label_class == prediction_class {
            hits += 1;
        }
        if i % 100 == 0 {
            reprint!(
                "Classifying...{:.2}%",
                (i as f32 / num_samples as f32) * 100.0
            );
        }
    }
    println!("\rClassifying done");

    println!(
        "{:.3}% accuracy out of {num_samples} test cases",
        (hits as f32 / num_samples as f32) * 100.0
    );

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
    TestMnist(TestMnistArgs),
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
    arch: Option<Vec<usize>>,

    #[clap(short, long)]
    save: PathBuf,

    #[clap(short = 'i', long, default_value_t = 1000)]
    steps: usize,

    #[clap(short, long)]
    save_every: Option<usize>,

    #[clap(short, long, default_value_t = 1e-7)]
    temperature: Precision,

    #[clap(short, long)]
    batch_size: Option<usize>,
}

#[derive(Parser)]
struct TestMnistArgs {
    model: PathBuf,

    #[clap(short = 'd', long, default_value = "mnist")]
    mnist_data: PathBuf,
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
        Command::TestMnist(args) => test_mnist(args),
        Command::TrainXor(args) => train_xor(args),
        Command::Inference(args) => inference(args),
    }
}
