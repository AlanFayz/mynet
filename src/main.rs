use std::fs::OpenOptions;
use std::io::BufWriter;
use std::io::Write;
use std::path::Path;
use std::time::Instant;

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
        network.train(&inputs, &outputs, 100.0);

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
    let (inputs, outputs) = EscData::parse("ESC-50-master/meta/esc50.csv")
        .unwrap()
        .load_data()
        .expect("failed to load data");

    let mut network = Network::read("sound-classifier.bin").unwrap_or(Network::new(vec![
        inputs.shape()[1],
        256,
        128,
        50usize,
    ]));

    let start_time = Instant::now();
    let mut save_timer: f64 = 0.0;

    loop {
        let delta_start = Instant::now();
        let (training_inputs, training_outputs) = sample_rows(&inputs, &outputs, 256);
        network.train(&training_inputs, &training_outputs, 0.1);

        let current_time = (Instant::now() - start_time).as_secs_f64();

        save_timer += (Instant::now() - delta_start).as_secs_f64();

        if save_timer >= 10.0 {
            let (testing_inputs, testing_outputs) = sample_rows(&inputs, &outputs, 100);
            let accuracy = network.calculate_accuracy(&testing_inputs, &testing_outputs);

            network.serialize("sound-classifier.bin");
            save_results_to_csv(current_time, accuracy);
            save_timer = 0.0;
        }
    }
}

fn main() {
    mnist_test();
}
