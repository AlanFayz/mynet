use std::collections::HashMap;
use std::fs::File;
use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::thread::current;
use std::time::Duration;
use std::time::Instant;

use hound::WavReader;
use ndarray::Array1;
use ndarray::Array2;
use ndarray::array;
use rand::seq::IteratorRandom;

use crate::esc_data::*;
use crate::mnist_data::*;
use crate::network::*;

mod esc_data;
mod mnist_data;
mod network;

extern crate blas_src;

fn get_or_insert_file<'a>(
    file_buffers: &'a mut HashMap<String, Vec<f32>>,
    file: &EscFile,
) -> &'a Vec<f32> {
    let name = file.name.clone();

    let entry = file_buffers.entry(name.clone());
    let buf_ref = entry.or_insert_with(|| {
        let full_path = Path::new("ESC-50-master/audio").join(&name);
        WavReader::open(full_path)
            .expect("failed to load file")
            .samples::<i16>()
            .step_by(10)
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect()
    });

    buf_ref
}

fn sample(
    training_data: &EscData,
    file_buffers: &mut HashMap<String, Vec<f32>>,
    sample_count: usize,
) -> (Array2<f32>, Array2<f32>) {
    let mut rng = rand::rng();

    let epoch: Vec<&EscFile> = training_data
        .mappings
        .iter()
        .choose_multiple(&mut rng, sample_count);

    let input_vec: Vec<Vec<f32>> = epoch
        .iter()
        .map(|file| get_or_insert_file(file_buffers, &file).clone())
        .collect();

    let inputs = Array2::from_shape_vec(
        (input_vec.len(), input_vec[0].len()),
        input_vec.into_iter().flatten().collect(),
    )
    .unwrap();

    let output_vec: Vec<Vec<f32>> = epoch
        .iter()
        .map(|file| {
            let mut ans = vec![0.0; 50];
            ans[file.target as usize] = 1.0;
            ans
        })
        .collect();

    let outputs = Array2::from_shape_vec(
        (output_vec.len(), output_vec[0].len()),
        output_vec.into_iter().flatten().collect(),
    )
    .unwrap();

    (inputs, outputs)
}

#[allow(dead_code)]
fn nand_test() {
    let mut network = Network::read("nand_network_training_test.bin")
        .unwrap_or(Network::new(vec![2usize, 1usize]));

    for _ in 0..100 {
        let inputs = array![[0.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 1.0]];

        let outputs = array![[1.0], [1.0], [1.0], [0.0]];

        network.train(&inputs, &outputs, 0.1);

        let test_cases: Vec<(Array1<f32>, &str)> = vec![
            (array![0.0, 0.0], "00"),
            (array![0.0, 1.0], "01"),
            (array![1.0, 0.0], "10"),
            (array![1.0, 1.0], "11"),
        ];

        for (input, label) in test_cases {
            network.set_input(&input.view());
            network.run();
            let result = network.get_output()[0];
            println!("{label}: {result:.6}");
        }

        println!("done");
    }

    network.serialize("nand_network_training_test.bin");
}

fn sample_rows(
    inputs: &Array2<f32>,
    outputs: &Array2<f32>,
    count: usize,
) -> (Array2<f32>, Array2<f32>) {
    let mut rng = rand::rng();
    let num_rows = inputs.shape()[0];

    let indices: Vec<usize> = (0..num_rows).choose_multiple(&mut rng, count);

    let sampled_inputs: Array2<f32> = Array2::from_shape_vec(
        (count, inputs.shape()[1]),
        indices
            .iter()
            .flat_map(|&i| inputs.row(i).to_owned().to_vec())
            .collect(),
    )
    .unwrap();

    let sampled_outputs: Array2<f32> = Array2::from_shape_vec(
        (count, outputs.shape()[1]),
        indices
            .iter()
            .flat_map(|&i| outputs.row(i).to_owned().to_vec())
            .collect(),
    )
    .unwrap();

    (sampled_inputs, sampled_outputs)
}

fn save_results_to_csv(time: f64, accuracy: f32) {
    let path = "results.csv";
    let file_exists = Path::new(path).exists();

    let file = OpenOptions::new()
        .create(true)      
        .append(true)      
        .open(path)
        .expect("Could not open file");

    let mut writer = BufWriter::new(file);

    if !file_exists {
        writeln!(writer, "time,accuracy").unwrap();
    }

    writeln!(writer, "{},{}", time, accuracy).unwrap();
}

#[allow(dead_code)]
fn mnist_test() {
    let training_images_filepath = "mnist/train-images-idx3-ubyte/train-images.idx3-ubyte";
    let training_labels_filepath = "mnist/train-labels-idx1-ubyte/train-labels.idx1-ubyte";
    let test_images_filepath = "mnist/t10k-images-idx3-ubyte/t10k-images.idx3-ubyte";
    let test_labels_filepath = "mnist/t10k-labels-idx1-ubyte/t10k-labels.idx1-ubyte";
    let network_name = "mnist_classifier.bin";

    let (train_images, train_labels, test_images, test_labels) = MnistDataLoader::new(
        training_images_filepath,
        training_labels_filepath,
        test_images_filepath,
        test_labels_filepath,
    )
    .load_data()
    .unwrap();

    let mut network = Network::read(network_name).unwrap_or(Network::new(vec![
        train_images.shape()[1],
        32,
        32,
        10,
    ]));

    let start_time = Instant::now();
    let mut save_timer: f64 = 0.0;
    
    loop {
        let delta_start = Instant::now();
        let (inputs, outputs) = sample_rows(&train_images, &train_labels, 256);
        network.train(&inputs, &outputs, 0.1);

        let current_time = (Instant::now() - start_time).as_secs_f64();

        save_timer += (Instant::now() - delta_start).as_secs_f64();
        
        if save_timer >= 10.0 {
            let (inputs, outputs) = sample_rows(&test_images, &test_labels, 100);
            let accuracy = network.calculate_accuracy(&inputs, &outputs);

            network.serialize(network_name);
            save_results_to_csv(current_time, accuracy);
            save_timer = 0.0;
        }
    }
}

#[allow(dead_code)]
fn esc50_test() {
    let mut file_buffers = HashMap::<String, Vec<f32>>::new();

    let mut network = Network::read("sound-classifier.bin")
        .unwrap_or(Network::new(vec![22050usize, 128usize, 50usize]));

    let training_data = EscData::parse("ESC-50-master/meta/esc50.csv").unwrap();

    for _ in 0..1 {
        let (inputs, outputs) = sample(&training_data, &mut file_buffers, 16usize);

        let prev = Instant::now();
        network.train(&inputs, &outputs, 0.01);
        let time_taken = (Instant::now() - prev).as_secs_f64();
        println!("Time taken: {time_taken}s");
    }

    let (inputs, outputs) = sample(&training_data, &mut file_buffers, 5);
    let accuracy = network.calculate_accuracy(&inputs, &outputs);

    println!("{accuracy}");

    network.serialize("sound-classifier.bin");
}

fn main() {
    mnist_test();
}
