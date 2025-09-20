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
    let mut network = Network::new_with_values(vec![
        Layer::new_with_values(vec![
            SigmoidNeuron::new_with_values(vec![-2.0], 0.0),
            SigmoidNeuron::new_with_values(vec![-2.0], 0.0),
        ]),
        Layer::new_with_values(vec![SigmoidNeuron::new_with_values(Vec::new(), 3.0)]),
    ]);

    network.set_input(vec![0f32, 1f32]);
    network.run();

    let output = network
        .get_output()
        .iter()
        .map(|n| n.to_string() + " ")
        .collect::<String>();

    println!("{output}");
}
