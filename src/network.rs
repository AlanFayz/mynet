use bincode;
use ndarray::{Array1, Array2, ArrayView1, Axis};
use rand_distr::Distribution;
use rayon::prelude::*;
use serde::{Deserialize, Serialize};
use std::fs::File;
use std::io::{Read, Write};

#[allow(dead_code)]
fn sigmoid(a: f32) -> f32 {
    1.0 / (1.0 + (-a).exp())
}

#[allow(dead_code)]
fn sigmoid_derivative(a: &Array1<f32>) -> Array1<f32> {
    a * (1.0 - a)
}

#[allow(dead_code)]
fn leaky_relu(x: f32) -> f32 {
    if x > 0.0 { x } else { 0.01 * x }
}

#[allow(dead_code)]
fn leaky_relu_derivative(a: &Array1<f32>) -> Array1<f32> {
    a.clone().mapv_into(|x| if x > 0.0 { 1.0 } else { 0.01 })
}

#[allow(dead_code)]
fn softmax(x: &Array1<f32>) -> Array1<f32> {
    let max = x.fold(f32::NEG_INFINITY, |a, &b| a.max(b));
    let exp_x: Array1<f32> = x.mapv(|v| (v - max).exp());
    &exp_x / exp_x.sum()
}

const ACTIVATION_FUNCTION: fn(f32) -> f32 = sigmoid;
const DERIVATIVE_FUNCTION: fn(&Array1<f32>) -> Array1<f32> = sigmoid_derivative;

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Layer {
    weights: Array2<f32>,
    biases: Array1<f32>,
    activations: Array1<f32>,
    deltas: Array1<f32>,
    weight_derivatives: Array2<f32>,
    bias_derivatives: Array1<f32>,
}

impl Layer {
    pub fn new(size: usize, previous_layer_size: usize) -> Self {
        let mut rng = rand::rng();
        let distribution = rand_distr::Normal::new(0.0, 1.0).unwrap();

        Self {
            weights: Array2::from_shape_fn((size, previous_layer_size), |_| {
                distribution.sample(&mut rng)
            }),
            biases: Array1::from_shape_fn(size, |_| distribution.sample(&mut rng)),
            activations: Array1::zeros(size),
            deltas: Array1::zeros(size),
            weight_derivatives: Array2::zeros((size, previous_layer_size)),
            bias_derivatives: Array1::zeros(size),
        }
    }

    pub fn feed_forward(&mut self, previous_layer: &Self) {
        self.activations = self.weights.dot(&previous_layer.activations) + &self.biases;
        self.activations.mapv_inplace(ACTIVATION_FUNCTION);
    }
}

#[derive(Serialize, Deserialize, Debug, Clone)]
pub struct Network {
    layers: Vec<Layer>,
}

impl Network {
    pub fn new(layer_sizes: Vec<usize>) -> Self {
        Self {
            layers: layer_sizes
                .iter()
                .enumerate()
                .map(|(index, layer_size)| {
                    Layer::new(
                        *layer_size,
                        if index == 0 {
                            0 as usize
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

        let decoded: Self = bincode::deserialize(&buffer).ok()?;
        Some(decoded)
    }

    pub fn get_output(&self) -> Array1<f32> {
        self.layers.last().unwrap().activations.clone()
    }

    pub fn set_input(&mut self, input_activations: &ArrayView1<f32>) {
        let input_layer = &mut self.layers[0];
        input_layer.activations = input_activations.into_owned();
    }

    pub fn run(&mut self) {
        for i in 1..self.layers.len() {
            let (left, right) = self.layers.split_at_mut(i);
            let prev_layer = &mut left[i - 1];
            let curr_layer = &mut right[0];
            curr_layer.feed_forward(prev_layer);
        }
    }

    fn step_derivatives(
        &mut self,
        training_input: &ArrayView1<f32>,
        expected_output: &ArrayView1<f32>,
    ) {
        self.set_input(training_input);
        self.run();

        let l = self.layers.len() - 1;
        let activation_derivatives = DERIVATIVE_FUNCTION(&self.layers[l].activations);

        let mut deltas = (&self.layers[l].activations - expected_output) * activation_derivatives;

        self.layers[l].deltas = deltas.clone();
        self.layers[l].bias_derivatives += &deltas;

        let delta_outer = self.layers[l]
            .deltas
            .view()
            .insert_axis(Axis(1))
            .dot(&self.layers[l - 1].activations.view().insert_axis(Axis(0)));

        self.layers[l].weight_derivatives += &delta_outer;

        for l in (1..self.layers.len() - 1).rev() {
            let activation_derivatives = DERIVATIVE_FUNCTION(&self.layers[l].activations);

            let sums = self.layers[l + 1].weights.t().dot(&deltas);

            deltas = sums * activation_derivatives;

            self.layers[l].deltas = deltas.clone();
            self.layers[l].bias_derivatives += &deltas;

            let delta_outer = self.layers[l]
                .deltas
                .view()
                .insert_axis(Axis(1))
                .dot(&self.layers[l - 1].activations.view().insert_axis(Axis(0)));

            self.layers[l].weight_derivatives += &delta_outer;
        }
    }

    fn zero_gradients(&mut self) {
        for l in 0..self.layers.len() {
            self.layers[l].weight_derivatives.fill(0.0);
            self.layers[l].bias_derivatives.fill(0.0);
        }
    }

    fn calculate_derivatives(
        &mut self,
        training_inputs: &Array2<f32>,
        expected_outputs: &Array2<f32>,
    ) {
        assert_eq!(training_inputs.shape()[0], expected_outputs.shape()[0]);
        self.zero_gradients();

        let batch_size = training_inputs.shape()[0] / num_cpus::get();

        let networks: Vec<Self> = training_inputs
            .axis_iter(Axis(0))
            .zip(expected_outputs.axis_iter(Axis(0)))
            .collect::<Vec<_>>()
            .par_chunks(batch_size)
            .map(|chunk| {
                let mut cloned = self.clone();

                for (input_row, expected_row) in chunk {
                    cloned.step_derivatives(&input_row.view(), &expected_row.view());
                }

                cloned
            })
            .collect();

        for l in 0..self.layers.len() {
            for network in &networks {
                self.layers[l].weight_derivatives += &network.layers[l].weight_derivatives;
                self.layers[l].bias_derivatives += &network.layers[l].bias_derivatives;
            }

            self.layers[l].weight_derivatives /= networks.len() as f32;
            self.layers[l].bias_derivatives /= networks.len() as f32;
        }
    }

    pub fn train(
        &mut self,
        training_inputs: &Array2<f32>,
        expected_outputs: &Array2<f32>,
        learning_rate: f32,
    ) {
        self.calculate_derivatives(training_inputs, expected_outputs);

        for l in 0..self.layers.len() {
            self.layers[l].weights = &self.layers[l].weights
                - (learning_rate / training_inputs.shape()[0] as f32)
                    * &self.layers[l].weight_derivatives;

            self.layers[l].biases = &self.layers[l].biases
                - (learning_rate / training_inputs.shape()[0] as f32)
                    * &self.layers[l].bias_derivatives;
        }
    }

    pub fn calculate_accuracy(&mut self, inputs: &Array2<f32>, outputs: &Array2<f32>) -> f32 {
        assert_eq!(inputs.shape()[0], outputs.shape()[0]);
        let mut total_correct = 0;

        for (input_row, expected_row) in inputs.axis_iter(Axis(0)).zip(outputs.axis_iter(Axis(0))) {
            self.set_input(&input_row.view());
            self.run();

            let predicted = self.get_output();

            let predicted_index = predicted
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            let expected_index = expected_row
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
                .map(|(idx, _)| idx)
                .unwrap();

            if predicted_index == expected_index {
                total_correct += 1;
            }
        }

        total_correct as f32 / inputs.shape()[0] as f32
    }
}
