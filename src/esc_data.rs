use std::{fs::File, io::Read};

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
}
