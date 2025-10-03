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
    pub bias_deriviative: f32,
    pub weights: Vec<f32>,
    pub weight_derivatives: Vec<f32>,
    pub delta: f32,
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
            delta: 0.0,
        }
    }

    pub fn set_activation(&mut self, activation: f32) {
        self.activation = 1.0 / (1.0 + (-activation).exp());
    }

    pub fn activation_derivative(&self) -> f32 {
        self.activation * (1.0 - self.activation)
    }
}

#[derive(Serialize, Deserialize, Debug)]
pub struct Layer {
    neurons: Vec<SigmoidNeuron>,
}

impl Layer {
    pub fn new(size: u64, previous_layer_size: u64) -> Self {
        Self {
            neurons: (0..size)
                .map(|_| SigmoidNeuron::new(previous_layer_size))
                .collect(),
        }
    }

    pub fn feed_forward(&mut self, previous_layer: &Self) {
        for n in 0..self.neurons.len() {
            let mut sum = 0.0;
            for w in 0..self.neurons[n].weights.len() {
                sum += self.neurons[n].weights[w] * previous_layer.neurons[w].activation;
            }

            let activation = sum + self.neurons[n].bias;
            self.neurons[n].set_activation(activation);
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
                        if index == 0 {
                            0u64
                        } else {
                            layer_sizes[index - 1]
                        },
                    )
                })
                .collect(),
        }
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
            layer2.feed_forward(layer1);
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

    fn step_derivatives(&mut self, training_input: &Vec<f32>, expected_output: &Vec<f32>) {
        self.set_input(training_input);
        self.run();

        let l = self.layers.len() - 1;

        for n in 0..self.layers[l].neurons.len() {
            let dc_da = self.layers[l].neurons[n].activation - expected_output[n];
            let activation_derivative = self.layers[l].neurons[n].activation_derivative();

            self.layers[l].neurons[n].delta = dc_da * activation_derivative;
            self.layers[l].neurons[n].bias_deriviative += self.layers[l].neurons[n].delta;

            for w in 0..self.layers[l].neurons[n].weights.len() {
                self.layers[l].neurons[n].weight_derivatives[w] +=
                    self.layers[l - 1].neurons[w].activation * self.layers[l].neurons[n].delta;
            }
        }

        for l in (1..self.layers.len() - 1).rev() {
            for n in 0..self.layers[l].neurons.len() {
                let activation_derivative = self.layers[l].neurons[n].activation_derivative();
                let mut sum = 0.0;

                for k in 0..self.layers[l + 1].neurons.len() {
                    sum += self.layers[l + 1].neurons[k].weights[n]
                        * self.layers[l + 1].neurons[k].delta;
                }

                self.layers[l].neurons[n].delta = sum * activation_derivative;
                self.layers[l].neurons[n].bias_deriviative += self.layers[l].neurons[n].delta;

                for w in 0..self.layers[l].neurons[n].weights.len() {
                    self.layers[l].neurons[n].weight_derivatives[w] +=
                        self.layers[l].neurons[n].delta * self.layers[l - 1].neurons[w].activation;
                }
            }
        }
    }

    fn zero_gradients(&mut self) {
        for l in 0..self.layers.len() {
            for n in 0..self.layers[l].neurons.len() {
                for w in 0..self.layers[l].neurons[n].weights.len() {
                    self.layers[l].neurons[n].weight_derivatives[w] = 0.0;
                }

                self.layers[l].neurons[n].bias_deriviative = 0.0;
            }
        }
    }

    fn average_and_length_gradients(&mut self, count: f32) -> f32 {
        let mut total: f32 = 0.0;

        for l in 0..self.layers.len() {
            for n in 0..self.layers[l].neurons.len() {
                for w in 0..self.layers[l].neurons[n].weights.len() {
                    self.layers[l].neurons[n].weight_derivatives[w] =
                        self.layers[l].neurons[n].weight_derivatives[w] / count;

                    total += self.layers[l].neurons[n].weight_derivatives[w];
                }

                self.layers[l].neurons[n].bias_deriviative =
                    self.layers[l].neurons[n].bias_deriviative / count;

                total += self.layers[l].neurons[n].bias_deriviative;
            }
        }

        return total.sqrt();
    }

    fn calculate_derivatives(
        &mut self,
        training_inputs: &Vec<Vec<f32>>,
        expected_outputs: &Vec<Vec<f32>>,
    ) -> f32 {
        assert_eq!(training_inputs.len(), expected_outputs.len());
        self.zero_gradients();

        for (input, output) in zip(training_inputs, expected_outputs) {
            self.step_derivatives(input, output);
        }

        return self.average_and_length_gradients(training_inputs.len() as f32);
    }

    pub fn train(
        &mut self,
        training_inputs: &Vec<Vec<f32>>,
        expected_outputs: &Vec<Vec<f32>>,
        learning_rate: f32,
    ) {
        let cost_derivative_length = self
            .calculate_derivatives(training_inputs, expected_outputs)
            .max(1e-6);

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

    pub fn calculate_accuracy(
        &mut self,
        inputs: &Vec<Vec<f32>>,
        outputs: &Vec<Vec<f32>>,
        threshold: f32,
    ) -> f32 {
        assert_eq!(inputs.len(), outputs.len());

        let mut total_correct = 0;
        for (input, output) in zip(inputs, outputs) {
            self.set_input(input);
            self.run();

            println!("--output--");
            for i in output {
                print!("{i} ");
            }
            println!();
            println!("--myOutput--");
            for i in self.get_output() {
                print!("actual {i} real {} ", if i >= threshold { 1 } else { 0 });
            }
            println!();
            println!("-------");

            total_correct += self
                .get_output()
                .iter()
                .map(|x| if *x >= threshold { 1.0 } else { 0.0 })
                .enumerate()
                .all(|(index, x)| x == output[index]) as i32;
        }

        return total_correct as f32 / inputs.len() as f32;
    }
}
