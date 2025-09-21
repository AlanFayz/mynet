use hound::WavReader;

use crate::network::*;

mod network;

fn main() {
    let mut reader = WavReader::open("ESC-50-master/audio/1-137-A-32.wav").unwrap();
    let samples: Vec<f32> = reader
        .samples::<i16>()
        .map(|s| s.unwrap() as f32 / i16::MAX as f32)
        .collect();

    println!("{} samples ready for NN input", samples.len());

    // simple nand gate for testing
    let mut network = Network::read("nand_network.bin").unwrap();

    let cost = network.total_cost(
        &vec![
            vec![0.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 0.0],
            vec![1.0, 1.0],
        ],
        &vec![vec![1.0], vec![1.0], vec![1.0], vec![0.0]],
    );
    println!("{cost}");
    println!("done");

    network.serialize("nand_network.bin");
}
