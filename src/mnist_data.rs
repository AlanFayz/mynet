use ndarray::prelude::*;
use std::{fs::File, io::Read};

pub struct MnistDataLoader {
    training_image_filepath: String,
    training_labels_filepath: String,
    test_image_filepath: String,
    test_labels_filepath: String,
}

impl MnistDataLoader {
    pub fn new(
        training_image_filepath: &str,
        training_labels_filepath: &str,
        test_image_filepath: &str,
        test_labels_filepath: &str,
    ) -> MnistDataLoader {
        MnistDataLoader {
            training_image_filepath: training_image_filepath.to_owned(),
            training_labels_filepath: training_labels_filepath.to_owned(),
            test_image_filepath: test_image_filepath.to_owned(),
            test_labels_filepath: test_labels_filepath.to_owned(),
        }
    }

    fn read_u32_be(&self, file: &mut File) -> Option<u32> {
        let mut buf = [0u8; 4];
        file.read_exact(&mut buf).ok()?;
        Some(u32::from_be_bytes(buf))
    }

    fn read_labels(&self, filepath: &str) -> Option<Vec<u8>> {
        let mut file = File::open(filepath).ok()?;
        let magic = self.read_u32_be(&mut file)?;
        let size = self.read_u32_be(&mut file)?;

        if magic != 2049 {
            return None;
        }

        let mut vec: Vec<u8> = vec![0u8; size as usize];
        file.read_exact(&mut vec).ok()?;

        Some(vec)
    }

    fn read_images(&self, filepath: &str) -> Option<(Vec<u8>, u32, u32)> {
        let mut file = File::open(filepath).ok()?;
        let magic = self.read_u32_be(&mut file)?;
        let size = self.read_u32_be(&mut file)?;
        let rows = self.read_u32_be(&mut file)?;
        let cols = self.read_u32_be(&mut file)?;

        if magic != 2051 {
            return None;
        }

        let size = size * rows * cols;

        let mut vec: Vec<u8> = vec![0u8; size as usize];
        file.read_exact(&mut vec).ok()?;

        Some((vec, rows, cols))
    }

    fn read_images_labels(
        &self,
        images_filepath: &str,
        labels_filepath: &str,
    ) -> Option<(Array2<f32>, Array2<f32>)> {
        let labels_data = self.read_labels(labels_filepath)?;
        let (images_data, rows, cols) = self.read_images(images_filepath)?;

        let mut labels: Array2<f32> = Array2::zeros((labels_data.len(), 10));
        for (index, &l) in labels_data.iter().enumerate() {
            labels[[index, l as usize]] = 1.0;
        }

        let image_size = rows * cols;

        let images: Array2<f32> = Array2::from_shape_vec(
            (labels_data.len(), image_size as usize),
            images_data.iter().map(|&x| (x as f32 / 128.0) - 1.0).collect(),
        )
        .unwrap();

        Some((images, labels))
    }

    pub fn load_data(&self) -> Option<(Array2<f32>, Array2<f32>, Array2<f32>, Array2<f32>)> {
        let (x_train, y_train) = self.read_images_labels(
            &self.training_image_filepath,
            &self.training_labels_filepath,
        )?;

        let (x_test, y_test) =
            self.read_images_labels(&self.test_image_filepath, &self.test_labels_filepath)?;

        Some((x_train, y_train, x_test, y_test))
    }
}
