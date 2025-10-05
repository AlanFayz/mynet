use std::{fs::File, io::Read, path::Path};

use hound::WavReader;
use ndarray::Array2;
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

    pub fn load_data(&self) -> Option<(Array2<f32>, Array2<f32>)> {
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

        Some((inputs, outputs))
    }
}
