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
    //let mut network = Network::read("nand_network.bin").unwrap();
    let mut network = Network::read("nand_network_training_test.bin").unwrap();

    for _ in 0..100 {
        network.train(
            &vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            &vec![vec![1.0], vec![1.0], vec![1.0], vec![0.0]],
            0.1,
        );

        let cost = network.total_cost(
            &vec![
                vec![0.0, 0.0],
                vec![0.0, 1.0],
                vec![1.0, 0.0],
                vec![1.0, 1.0],
            ],
            &vec![vec![1.0], vec![1.0], vec![1.0], vec![0.0]],
        );

        println!("cost: {cost}");

        let test_cases = vec![
            (vec![0.0, 0.0], "00"),
            (vec![0.0, 1.0], "01"),
            (vec![1.0, 0.0], "10"),
            (vec![1.0, 1.0], "11"),
        ];

        for (input, label) in test_cases {
            network.set_input(&input);
            network.run();
            let result = network.get_output()[0];
            println!("{label}: {result:.6}");
        }

        println!("done");
    }

    network.serialize("nand_network_training_test.bin");
}
