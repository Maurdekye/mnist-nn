#![feature(array_windows)]
#![feature(iterator_try_collect)]

use std::{
    error::Error,
    fs::File,
    io::{self, BufReader, Read},
    path::PathBuf,
    time::Duration,
};

use clap::{Parser, Subcommand, ValueEnum};

mod nn;

use csv::Writer;
use itertools::Itertools;
use nn::*;
use progress_observer::Observer;
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};
use serde::Serialize;
use std::fmt::Display;

fn read_u32<R: Read>(reader: &mut R) -> Result<u32, Box<dyn Error>> {
    let mut uint = [0; 4];
    reader.read_exact(&mut uint)?;
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

    Ok(images
        .bytes()
        .chunks((width * height) as usize)
        .into_iter()
        .take(num_items as usize)
        .map(|chunk| {
            chunk
                .map(|opacity| Ok::<_, io::Error>((opacity? as Precision) / 255.0))
                .try_collect()
        })
        .try_collect()?)
}

fn load_mnist_labels(path: PathBuf) -> Result<Vec<Vec<Precision>>, Box<dyn Error>> {
    let mut labels = BufReader::new(File::open(path)?);
    if read_u32(&mut labels)? != 0x00000801 {
        Err("Magic number does not match on labels file")?;
    }
    let num_items = read_u32(&mut labels)?;

    labels
        .bytes()
        .take(num_items as usize)
        .map(|label| {
            let mut label_vector = vec![0.0; 10];
            label_vector[label? as usize] = 1.0;
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

fn load_mnist_training(base_path: PathBuf) -> Result<Vec<Sample>, Box<dyn Error>> {
    load_mnist(
        base_path.join("train-images.idx3-ubyte"),
        base_path.join("train-labels.idx1-ubyte"),
    )
}

fn load_mnist_testing(base_path: PathBuf) -> Result<Vec<Sample>, Box<dyn Error>> {
    load_mnist(
        base_path.join("t10k-images.idx3-ubyte"),
        base_path.join("t10k-labels.idx1-ubyte"),
    )
}

#[derive(Serialize)]
struct TrainingLogRecord {
    iteration: usize,
    loss: Precision,
}

fn train_mnist(args: TrainMnistArgs) -> Result<(), Box<dyn Error>> {
    let mut logfile = args
        .logfile
        .map(|logfile| Writer::from_path(logfile))
        .transpose()?;

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
    let samples = load_mnist_training(args.mnist_data)?;

    println!("Beginning training");

    let mut observer = Observer::new(Duration::from_secs_f32(0.5));
    train::<SiLu, Sigmoid>(
        &mut model,
        &samples,
        args.steps,
        args.temperature,
        args.batch_size,
        |TrainingProgress {
             iteration,
             model,
             gradient: _,
             loss,
         }| {
            if observer.tick() {
                println!("iteration {iteration}: loss {loss}");
            }
            if let Some(logfile) = &mut logfile {
                logfile
                    .serialize(TrainingLogRecord { iteration, loss })
                    .expect("Error saving to log file");
                logfile.flush().unwrap();
            }
            if let Some(save_every) = args.save_every {
                if iteration > 0 && iteration % save_every == 0 {
                    println!("Saving checkpoint");
                    let mut filename = args.save.clone();
                    filename.set_file_name(format!(
                        "{iteration}-{}",
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
    train::<SiLu, Sigmoid>(
        &mut model,
        &samples,
        args.steps,
        args.temperature,
        None,
        |_| {},
    );
    println!("Training finished");

    println!("Saving model");
    save(&model, args.save)?;

    println!("Done");

    Ok(())
}

fn classify_label(output: &Vec<Precision>) -> usize {
    output
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.total_cmp(b))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

fn test_mnist(args: TestMnistArgs) -> Result<(), Box<dyn Error>> {
    let model = load(args.model)?;

    println!("Loading testing data");
    let samples = load_mnist_testing(args.mnist_data)?;
    let num_samples = samples.len();

    println!("Classifying...");
    let hits = samples
        .par_iter()
        .filter(|Sample(features, label)| {
            let prediction = model.inference::<SiLu, Sigmoid>(features);
            let label_class = classify_label(label);
            let prediction_class = classify_label(&prediction);
            label_class == prediction_class
        })
        .count();
    println!("\rClassifying done");

    println!(
        "{:.3}% accuracy out of {num_samples} test cases",
        (hits as f32 / num_samples as f32) * 100.0
    );

    Ok(())
}

fn visualize_mnist(args: VisualizeMnistArgs) -> Result<(), Box<dyn Error>> {
    let dataset = match args.datatset {
        DatasetType::Train => load_mnist_training(args.mnist_data),
        DatasetType::Test => load_mnist_testing(args.mnist_data),
    }?;

    let Some(Sample(features, label)) = dataset.get(args.sample) else {
        Err(format!("Sample '{}' out of range", args.sample))?
    };

    println!("{} sample #{}:", args.datatset, args.sample);
    println!("Features:");
    for row in features.chunks(28) {
        println!(
            "{}",
            row.iter()
                .map(|&cell| match cell {
                    _ if cell > 0.75 => "#",
                    _ if cell > 0.50 => "+",
                    _ if cell > 0.25 => ".",
                    _ => " ",
                })
                .collect::<Vec<_>>()
                .join("")
        );
    }

    let label = classify_label(label);
    println!("Label: {label}");

    Ok(())
}

fn inference(args: InferenceArgs) -> Result<(), Box<dyn Error>> {
    let model = load(args.model)?;
    println!("{:?}", args.input);
    let output = model.inference::<SiLu, Sigmoid>(&args.input);
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
    VisualizeMnist(VisualizeMnistArgs),
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

    #[clap(short, long)]
    logfile: Option<PathBuf>,
}

#[derive(Parser)]
struct TestMnistArgs {
    model: PathBuf,

    #[clap(short = 'd', long, default_value = "mnist")]
    mnist_data: PathBuf,
}

#[derive(ValueEnum, Clone)]
enum DatasetType {
    Train,
    Test,
}

impl Display for DatasetType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.to_possible_value().unwrap().get_name())
    }
}

#[derive(Parser)]
struct VisualizeMnistArgs {
    #[clap(short = 'd', long, default_value = "mnist")]
    mnist_data: PathBuf,

    datatset: DatasetType,

    sample: usize,
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
        Command::VisualizeMnist(args) => visualize_mnist(args),
        Command::TrainXor(args) => train_xor(args),
        Command::Inference(args) => inference(args),
    }
}
