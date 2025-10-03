use std::collections::HashMap;
use std::path::Path;
use std::time::Instant;

use hound::WavReader;
use rand::rng;
use rand::seq::IteratorRandom;

use crate::esc_data::*;
use crate::network::*;

mod esc_data;
mod network;

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
            .map(|s| s.unwrap() as f32 / i16::MAX as f32)
            .collect()
    });

    buf_ref
}

fn main() {
    let mut file_buffers = HashMap::<String, Vec<f32>>::new();
    let mut network = Network::read("sound-classifier.bin").unwrap_or(Network::new(vec![22500u64, 128u64, 50u64]));
    let training_data = EscData::parse("ESC-50-master/meta/esc50.csv").unwrap();
    let mut rng = rng();

    for _ in 0..1 {
        let epoch: Vec<&EscFile> = training_data.mappings.iter().choose_multiple(&mut rng, 16);
        let inputs: Vec<Vec<f32>> = epoch
            .iter()
            .map(|file| get_or_insert_file(&mut file_buffers, &file).clone())
            .collect();

        let outputs: Vec<Vec<f32>> = epoch
            .iter()
            .map(|file| {
                let mut ans = Vec::new();
                ans.resize(50usize, 0.0);
                ans[file.target as usize] = 1.0;
                ans
            })
            .collect();

        let prev = Instant::now();
        network.train(&inputs, &outputs, 2.0);
        let time_taken = (Instant::now() - prev).as_secs_f64();
        println!("Time taken: {time_taken}s");
    }

    let epoch: Vec<&EscFile> = training_data.mappings.iter().choose_multiple(&mut rng, 5);
    let inputs: Vec<Vec<f32>> = epoch
        .iter()
        .map(|file| get_or_insert_file(&mut file_buffers, &file).clone())
        .collect();

    let outputs: Vec<Vec<f32>> = epoch
        .iter()
        .map(|file| {
            let mut ans = Vec::new();
            ans.resize(50usize, 0.0);
            ans[file.target as usize] = 1.0;
            ans
        })
        .collect();

    let accuracy = network.calculate_accuracy(&inputs, &outputs, 0.5);
    let cost = network.total_cost(&inputs, &outputs);
    println!("{accuracy}");
    println!("{cost}");
    network.serialize("sound-classifier.bin");
}
