use bincode;
use rand::prelude::*;
use serde::{Deserialize, Serialize};
use std::f32::EPSILON;
use std::fs::File;
use std::io::{Read, Write};
use std::iter::zip;

#[derive(Serialize, Deserialize, Debug)]
pub struct SigmoidNeuron {
    pub activation: f32,
    pub bias: f32,
    pub bias_deriviative: f32,
    pub weights: Vec<f32>,
    pub weight_derivatives: Vec<f32>,
}

impl SigmoidNeuron {
    pub fn new(size: u64) -> Self {
        let mut rng = rand::rng();

        Self {
            activation: 0.0,
            bias: rng.random(),
            bias_deriviative: 0.0,
            weights: (0..size).map(|_| rng.random::<f32>()).collect(),
            weight_derivatives: (0..size).map(|_| 0.0).collect(),
        }
    }

    pub fn new_with_values(weights: &Vec<f32>, bias: f32) -> Self {
        Self {
            activation: 0.0,
            bias,
            bias_deriviative: 0.0,
            weights: weights.clone(),
            weight_derivatives: Vec::with_capacity(weights.len()),
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

        let mut decoded: Self = bincode::deserialize(&buffer).ok()?;
        for l in 0..decoded.layers.len() {
            for n in 0..decoded.layers[l].neurons.len() {
                let decoded_layer_neuron = &mut decoded.layers[l].neurons[n];
                decoded_layer_neuron
                    .weight_derivatives
                    .resize(decoded_layer_neuron.weights.len(), 0.0);
            }
        }

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

    fn calculate_derivatives(
        &mut self,
        training_inputs: &Vec<Vec<f32>>,
        expected_outputs: &Vec<Vec<f32>>,
    ) -> f32 {
        const H: f32 = 1e-4;
        let fx = self.total_cost(training_inputs, expected_outputs);
        let mut total = 0.0;

        for l in 0..self.layers.len() {
            for n in 0..self.layers[l].neurons.len() {
                for w in 0..self.layers[l].neurons[n].weights.len() {
                    self.layers[l].neurons[n].weights[w] += H;
                    let fxh = self.total_cost(training_inputs, expected_outputs);
                    self.layers[l].neurons[n].weights[w] -= H;

                    let derivative = (fxh - fx) / H;
                    self.layers[l].neurons[n].weight_derivatives[w] = derivative;
                    total += derivative * derivative;
                }

                self.layers[l].neurons[n].bias += H;
                let fxh = self.total_cost(training_inputs, expected_outputs);
                self.layers[l].neurons[n].bias -= H;

                let derivative = (fxh - fx) / H;
                self.layers[l].neurons[n].bias_deriviative = derivative;
                total += derivative * derivative;
            }
        }

        return total.sqrt();
    }

    pub fn train(
        &mut self,
        training_inputs: &Vec<Vec<f32>>,
        expected_outputs: &Vec<Vec<f32>>,
        learning_rate: f32,
    ) {
        let cost_derivative_length = self.calculate_derivatives(training_inputs, expected_outputs);

        for l in 0..self.layers.len() {
            for n in 0..self.layers[l].neurons.len() {
                for w in 0..self.layers[l].neurons[n].weights.len() {
                    let weight = self.layers[l].neurons[n].weights[w];
                    let weight_derivative = self.layers[l].neurons[n].weight_derivatives[w];

                    self.layers[l].neurons[n].weights[w] =
                        weight - (learning_rate / cost_derivative_length) * weight_derivative;
                }

                let bias = self.layers[l].neurons[n].bias;
                let bias_derivative = self.layers[l].neurons[n].bias_deriviative;
                self.layers[l].neurons[n].bias =
                    bias - (learning_rate / cost_derivative_length) * bias_derivative;
            }
        }
    }
}
