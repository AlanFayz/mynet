use bincode;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};
use std::iter::zip;

#[derive(Serialize, Deserialize, Debug)]
pub struct SigmoidNeuron {
    pub activation: f32,
    pub bias: f32,
    pub weights: Vec<f32>,
}

impl SigmoidNeuron {
    pub fn new(size: u64) -> Self {
        let mut rng = rand::rng();

        Self {
            activation: rng.random(),
            bias: rng.random(),
            weights: (0..size).map(|_| rng.random::<f32>()).collect(),
        }
    }

    pub fn new_with_values(weights: Vec<f32>, bias: f32) -> Self {
        Self {
            activation: 0.0,
            bias,
            weights,
        }
    }

    pub fn set_activation(&mut self, activation: f32) {
        self.activation = 1.0 / (1.0 + (-activation).exp());
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Layer {
    neurons: Vec<SigmoidNeuron>,
}

impl Layer {
    pub fn new(size: u64, next_layer_size: u64) -> Self {
        Self {
            neurons: (0..size)
                .map(|_| SigmoidNeuron::new(next_layer_size))
                .collect(),
        }
    }

    pub fn new_with_values(neurons: Vec<SigmoidNeuron>) -> Self {
        Self { neurons }
    }

    pub fn feed_forward(&self, next_layer: &mut Self) {
        for (index, neuron) in next_layer.neurons.iter_mut().enumerate() {
            let new_activiation = self
                .neurons
                .iter()
                .map(|other_neuron| other_neuron.weights[index] * other_neuron.activation)
                .sum::<f32>()
                + neuron.bias;

            neuron.set_activation(new_activiation);
        }
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layer_sizes: Vec<u64>) -> Self {
        Self {
            layers: layer_sizes
                .iter()
                .enumerate()
                .map(|(index, layer_size)| {
                    Layer::new(
                        *layer_size,
                        if index + 1 < layer_sizes.len() {
                            layer_sizes[index + 1]
                        } else {
                            0u64
                        },
                    )
                })
                .collect(),
        }
    }

    pub fn new_with_values(layers: Vec<Layer>) -> Self {
        Self { layers }
    }

    pub fn serialize(&self, filename: &str) {
        let encoded: Vec<u8> = bincode::serialize(&self).unwrap();

        let mut file = File::create(filename).unwrap();
        file.write_all(&encoded).unwrap();
    }

    pub fn read(filename: &str) -> Option<Self> {
        let mut file = File::open(filename).ok()?;
        let mut buffer = Vec::new();
        file.read_to_end(&mut buffer).ok()?;

        let decoded: Self = bincode::deserialize(&buffer).ok()?;
        return Some(decoded);
    }

    pub fn get_output(&self) -> Vec<f32> {
        self.layers
            .last()
            .unwrap()
            .neurons
            .iter()
            .map(|neuron| neuron.activation)
            .collect()
    }

    pub fn set_input(&mut self, input_activations: &Vec<f32>) {
        let input_layer = &mut self.layers[0];

        for (index, neuron) in input_layer.neurons.iter_mut().enumerate() {
            neuron.activation = input_activations[index];
        }
    }

    pub fn run(&mut self) {
        for i in 0..self.layers.len() - 1 {
            let (left, right) = self.layers.split_at_mut(i + 1);
            let layer1 = &mut left[i];
            let layer2 = &mut right[0];
            layer1.feed_forward(layer2);
        }
    }

    pub fn total_cost(
        &mut self,
        training_inputs: &Vec<Vec<f32>>,
        expected_outputs: &Vec<Vec<f32>>,
    ) -> f32 {
        assert!(training_inputs.len() == expected_outputs.len());

        let mut total = 0.0;
        for (input, expect_output) in zip(training_inputs, expected_outputs) {
            self.set_input(input);
            self.run();

            let output = self.get_output();
            assert!(output.len() == expect_output.len());

            let result = output
                .iter()
                .enumerate()
                .map(|(index, activiation)| *activiation - expect_output[index])
                .fold(0.0, |acc, x| acc + x * x);

            total += result;
        }

        return total / (2.0 * training_inputs.len() as f32);
    }

    pub fn train(&mut self, training_inputs: &Vec<Vec<f32>>, expected_outputs: &Vec<Vec<f32>>) {
        unimplemented!()
    }
}
