use std::{fs::File, io::Read, path::Path};

use hound::WavReader;
use ndarray::{Array1, Array2, Axis, IntoDimension};
use rayon::iter::{IntoParallelRefIterator, ParallelIterator};

#[allow(dead_code)]
pub struct EscFile {
    pub name: String,
    pub fold: i32,
    pub target: i32,
    pub category: String,
}

pub struct EscData {
    pub mappings: Vec<EscFile>,
}

impl EscData {
    pub fn parse(filename: &str) -> Option<EscData> {
        let mut csv_file = File::open(filename).ok()?;
        let mut buffer = String::new();
        csv_file.read_to_string(&mut buffer).ok()?;

        Some(EscData {
            mappings: buffer
                .lines()
                .skip(1)
                .map(|line| {
                    let line_data: Vec<&str> = line.split(",").collect();

                    EscFile {
                        name: line_data[0].to_owned(),
                        fold: line_data[1].parse::<i32>().unwrap(),
                        target: line_data[2].parse::<i32>().unwrap(),
                        category: line_data[3].to_owned(),
                    }
                })
                .collect(),
        })
    }

    fn load_file(&self, path: &Path) -> Option<Vec<f32>> {
        Some(
            WavReader::open(path)
                .ok()?
                .samples::<i16>()
                .step_by(100)
                .map(|s| (s.unwrap() as f32 / i16::MAX as f32) * 2.0 - 1.0)
                .collect(),
        )
    }

    #[allow(dead_code)]
    fn compress_data_downsample_direct(&self, audio_sample: &Array2<f32>) -> Array2<f32> {
        const TARGET_DIM: (usize, usize) = (200, 200);

        let audio_size = audio_sample.shape()[1];
        let audio_width = (audio_size as f32).sqrt() as usize;
        let audio_height = audio_size / audio_width + audio_size % audio_width;

        let downsampled_audio: Vec<Array1<f32>> = audio_sample
            .axis_iter(Axis(0))
            .map(|audio| {
                let mut downsampled = Array1::zeros(TARGET_DIM.0 * TARGET_DIM.1);

                for y in 0..TARGET_DIM.1 {
                    for x in 0..TARGET_DIM.0 {
                        let scaled_x = (x as f32 / TARGET_DIM.0 as f32) * audio_width as f32;
                        let scaled_x = scaled_x as usize;

                        let scaled_y = (y as f32 / TARGET_DIM.1 as f32) * audio_height as f32;
                        let scaled_y = scaled_y as usize;

                        let index = x + y * TARGET_DIM.0;
                        let scaled_index = scaled_x + scaled_y * audio_width;

                        if index < downsampled.shape()[0] && scaled_index < audio.shape()[0] {
                            downsampled[index] = audio[scaled_index];
                        }
                    }
                }

                downsampled
            })
            .collect();

        Array2::from_shape_vec(
            (downsampled_audio.len(), downsampled_audio[0].len()),
            downsampled_audio.into_iter().flatten().collect(),
        )
        .unwrap()
    }

    pub fn load_data(&self) -> Option<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>)> {
        let base_path = Path::new("ESC-50-master/audio");
        let input_vec: Vec<Vec<f32>> = self
            .mappings
            .par_iter()
            .map(|file| {
                self.load_file(&base_path.join(&file.name))
                    .expect("failed to load file")
            })
            .collect();

        let inputs = Array2::from_shape_vec(
            (input_vec.len(), input_vec[0].len()),
            input_vec.into_iter().flatten().collect(),
        )
        .unwrap();

        let output_vec: Vec<Vec<f32>> = self
            .mappings
            .par_iter()
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

        let inputs = self.compress_data_downsample_direct(&inputs);

        let (training_data, testing_data) = inputs.view().split_at(Axis(0), inputs.shape()[0] / 2);

        let (training_labels, testing_labels) =
            outputs.view().split_at(Axis(0), inputs.shape()[0] / 2);

        Some((
            training_data.to_owned(),
            training_labels.to_owned(),
            testing_data.to_owned(),
            testing_labels.to_owned(),
        ))
    }
}
