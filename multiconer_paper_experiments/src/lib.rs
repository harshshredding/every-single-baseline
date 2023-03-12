use std::collections::HashMap;
use std::{error::Error, io, process};

use csv::WriterBuilder;
use serde_json::Value;
use std::fs::File;
use std::fs;

pub fn add(left: usize, right: usize) -> usize {
    left + right
}

pub fn create_some_string() -> String {
    return String::from("hello")
}

#[derive(Debug)]
pub struct Prediction {
    sample_id: String,
    start: i32,
    end: i32,
    entity_type: String, 
    extraction: String
}


pub fn get_token_offsets(samples_file_path: &str) -> HashMap<String, Vec<(i64, i64)>> {
    let samples_json_raw = fs::read_to_string(samples_file_path).unwrap();
    let all_samples: Vec<HashMap<String, Value>> = serde_json::from_str(&samples_json_raw)
        .expect("Cannot parse json into a vec of hashmap");
    //println!("{:?}", all_samples.get(0).expect("cannot get first sample"));
    let first_samples_id = all_samples[0]["id"].as_str().expect("cannot get the id");

    let mut token_offsets_dict = HashMap::new();
    for sample in all_samples { 
        let sample_id = sample["id"].as_str().expect(&format!("unable to parse sample_id: {:?}", sample["id"])).to_owned();
        let sample_tokens: Vec<HashMap<String, Value>> = serde_json::from_value(sample["annos"]["external"].clone()).expect("failed to extract token list");
        let tokens_offsets: Vec<(i64, i64)> = sample_tokens.into_iter()
            .map(|token| {
                let start = token["begin_offset"].as_i64().expect("could not parse start");
                let end = token["end_offset"].as_i64().expect("could not parse start");
                return (start, end);
            }).collect();
        token_offsets_dict.insert(sample_id, tokens_offsets);
    }
    return token_offsets_dict;
}

pub fn read_tsv_file(predictions_file_path: &str) -> Vec<Prediction> {
    let file = File::open(predictions_file_path)
        .expect(format!("Not able to open file:{}", predictions_file_path).as_str());
    let mut tsv_reader = csv::ReaderBuilder::new().delimiter(b'\t').from_reader(file);
    return tsv_reader.records()
        .map(|record| {
            let record = record.unwrap();
            let sample_id = record.get(0).expect(format!("cannot get sample id out {:?}", record).as_str());
            let start = record.get(1).expect(format!("cannot get start out {:?}", record).as_str());
            let end = record.get(2).expect(format!("cannot get end out {:?}", record).as_str());
            let entity_type = record.get(3).expect(format!("cannot get entity type out {:?}", record).as_str());
            let extraction = record.get(4).expect(&format!("cannot get extraction out {:?}", record));
            return Prediction {
                sample_id: sample_id.into(),
                start: start.parse().expect("could not parse start into int"),
                end: end.parse().expect("could not parse end into int"),
                entity_type: entity_type.to_string(),
                extraction: extraction.to_string()
            }
        }).collect()
}

pub fn get_predictions_dict(predictions_file_path: &str) -> HashMap<String, Vec<Prediction>> {
    let predictions = read_tsv_file(predictions_file_path);
    let mut predictions_dict = HashMap::new();
    for prediction in predictions {
        predictions_dict.entry(prediction.sample_id.clone())
            .or_insert_with(Vec::new).push(prediction);
    }
    return predictions_dict;
}

pub fn find_sub_token_predictions(predictions_file_path: &str, samples_file_path: &str, output_file_path: &str) {
    let mut writer = WriterBuilder::new().delimiter(b'\t').from_path(output_file_path)
                        .expect("unable to build writer");
    writer.write_record(&["sample_id", "begin", "end", "type", "extraction"]).expect("couldn't write header.");
    let token_offsets = get_token_offsets(samples_file_path);
    let predictions_dict = get_predictions_dict(predictions_file_path);
    let mut num_subword_predictions = 0;
    let mut num_predictions = 0;
    let mut sub_word_predictions = vec![];
    for (sample_id, predictions) in predictions_dict {
        let sample_token_offsets = token_offsets.get(&sample_id)
                                        .expect(&format!("failed to get token offsets for sample: {}", sample_id));
        for prediction in predictions {
            num_predictions += 1;
            let token_matching_start = sample_token_offsets.iter()
                .find(|offset| {
                    return offset.0 == (prediction.start as i64)
                });
            let token_matching_end = sample_token_offsets.iter()
                .find(|offset| {
                    return offset.1 == (prediction.end as i64)
                });
            if token_matching_start.is_none() || token_matching_end.is_none() {
                num_subword_predictions += 1;
                sub_word_predictions.push(prediction)
            } else {
                // write to file
                writer.serialize((prediction.sample_id, prediction.start, prediction.end, prediction.entity_type, prediction.extraction))
                    .expect("unable to write record to tsv file")
            }
        }
        writer.flush().expect("unable to flush")
    }
    println!("num subword predictions {}", num_subword_predictions);
    println!("num predictions {}", num_predictions); 
}


pub fn write_aligned_predictions(predictions_file_path: &str, samples_file_path: &str, output_file_path: &str) {
    let mut writer = WriterBuilder::new().delimiter(b'\t').from_path(output_file_path)
                        .expect("unable to build writer");
    writer.write_record(&["sample_id", "begin", "end", "type", "extraction"]).expect("couldn't write header.");
    let token_offsets = get_token_offsets(samples_file_path);
    let predictions_dict = get_predictions_dict(predictions_file_path);
    for (sample_id, predictions) in predictions_dict {
        let sample_token_offsets = token_offsets.get(&sample_id)
                                        .expect(&format!("failed to get token offsets for sample: {}", sample_id));
        for prediction in predictions {
            let token_surrounding_start = sample_token_offsets.iter()
                .find(|token_span| {
                    return (token_span.0 < (prediction.start as i64)) 
                        && ((prediction.start as i64) < token_span.1)
                });
            let token_surrounding_end = sample_token_offsets.iter()
                .find(|token_span| { 
                    return (token_span.0 < (prediction.end as i64)) 
                        && ((prediction.end as i64) < token_span.1)
                });
            let new_prediction_start = match token_surrounding_start {
                Some(token_span) => token_span.0,
                None => prediction.start as i64
            }; 
            let new_prediction_end = match token_surrounding_end {
                Some(token_span) => token_span.1,
                None => prediction.end as i64
            };
            writer.serialize((prediction.sample_id, new_prediction_start, new_prediction_end, prediction.entity_type, prediction.extraction))
                    .expect("unable to write record to tsv file")
        }
        writer.flush().expect("unable to flush")
    }
}

pub fn read_tokens() -> HashMap<String, Vec<(i32, i32)>> {
	let samples_file_path = "/home/harsh/every-single-baseline/preprocessed_data/multiconer_fine_valid_tokens_production_samples.json";
    unimplemented!()
}

#[cfg(test)]
mod tests {

    use serde_json::from_value;
    use serde_json::from_str;

    use super::*;

    #[test]
    fn it_works() {
        let result = add(2, 2);
        assert_eq!(result, 4);
    }

    #[test]
    fn create_some_string_works() {
        let result = create_some_string();
        assert_eq!(result, "hello")
    }

    #[test]
    fn read_a_tsv_file() {
        let seq_predictions_file_path = "/home/harsh/every-single-baseline/multiconer_paper_data/experiment_multiconer_with_test_gold_labels_multiconer_fine_vanilla_model_seq_large_default_valid_epoch_10_predictions.tsv";
        let predictions = read_tsv_file(seq_predictions_file_path);
        let first_prediction = predictions.iter().next().expect("could not get first prediction");
        assert_eq!(first_prediction.sample_id, "5239d808-f300-46ea-aa3b-5093040213a3", "sample id doesn't match");
        assert_eq!(first_prediction.start, 0, "begin doesn't match");
        assert_eq!(first_prediction.end, 9, "end doesn't match");
        assert_eq!(predictions.len(), 1326)
    }

    #[test]
    fn get_predictions_dict_test() {
        let seq_predictions_file_path = "/home/harsh/every-single-baseline/multiconer_paper_data/experiment_multiconer_with_test_gold_labels_multiconer_fine_vanilla_model_seq_large_default_valid_epoch_10_predictions.tsv";
        let predictions_dict = get_predictions_dict(seq_predictions_file_path);
        let predictions = predictions_dict.get("370fc8f3-20d6-4663-ad6f-796c44377008").expect("sample id should exist");
        assert_eq!(predictions.len(), 3)
    }


    #[test]
    fn serde_power() {
        let data = r#"
                {
                    "name": "John Doe",
                    "age": 43,
                    "phones": [
                        "+44 1234567",
                        "+44 2345678"
                    ]
                }"#;
        let person: HashMap<String, Value> = from_str(data)
                                                .expect("unable to parse into map of values");
        let phone_numbers: Vec<String> = from_value(person["phones"].to_owned())
                                                .expect("unable to parse into a vector of strings");
        assert_eq!(phone_numbers.len(), 2);
        assert_eq!(phone_numbers[0], "+44 1234567");
        assert_eq!(phone_numbers[1], "+44 2345678");
        assert_eq!(person.len(), 3);
    }


    #[test]
    fn test_get_token_offsets() {
        let samples_file_path = "/home/harsh/every-single-baseline/preprocessed_data/multiconer_fine_valid_tokens_production_samples.json";
        let token_offsets = get_token_offsets(samples_file_path);
        let tokens_for_sample = token_offsets.get("5239d808-f300-46ea-aa3b-5093040213a3").expect("id should have existed");
    }

    #[test]
    fn test_find_sub_token_predictions_valid() { 
        let samples_file_path = "/home/harsh/every-single-baseline/preprocessed_data/multiconer_fine_valid_tokens_production_samples.json";
        let seq_predictions_file_path = "/home/harsh/every-single-baseline/multiconer_paper_data/experiment_multiconer_with_test_gold_labels_multiconer_fine_vanilla_model_seq_large_default_valid_epoch_10_predictions.tsv";
        let seq_tsv_out_path = "./seq_without_boundary_errors_valid.tsv";
        let span_tsv_out_path = "./span_without_boundary_errors_valid.tsv";
        find_sub_token_predictions(seq_predictions_file_path, samples_file_path, seq_tsv_out_path);
        let span_predictions_file_path = "/home/harsh/every-single-baseline/multiconer_paper_data/experiment_multiconer_with_test_gold_labels_multiconer_fine_vanilla_span_large_default_valid_epoch_9_predictions.tsv";
        find_sub_token_predictions(span_predictions_file_path, samples_file_path, span_tsv_out_path);
    }

    #[test]
    #[ignore]
    fn test_find_sub_token_predictions_test_new() {
        let span_predictions_file_path = "/home/harsh/every-single-baseline/submission/no_special_tokens_predictions/experiment_multiconer_with_test_gold_labels_multiconer_fine_vanilla_span_large_default_test_epoch_7_predictions.tsv";
        let seq_predictions_file_path = "/home/harsh/every-single-baseline/submission/no_special_tokens_predictions/experiment_multiconer_with_test_gold_labels_multiconer_fine_vanilla_model_seq_large_default_test_epoch_7_predictions.tsv";
        let samples_file_path = "/home/harsh/every-single-baseline/preprocessed_data/multiconer_fine_test_tokens_production_samples.json";
        let seq_tsv_out_path = "./seq_without_boundary_errors_test.tsv";
        let span_tsv_out_path = "./span_without_boundary_errors_test.tsv";
        println!("TEST Seq");
        find_sub_token_predictions(seq_predictions_file_path, samples_file_path, seq_tsv_out_path);
        println!("TEST Span");
        find_sub_token_predictions(span_predictions_file_path, samples_file_path, span_tsv_out_path);
    }


    #[test]
    #[ignore]
    fn test_find_sub_token_predictions_test_old() {
        let seq_predictions_file_path = "/home/harsh/every-single-baseline/submission/old_predictions_test/experiment_multiconer_with_test_gold_labels_old_multiconer_fine_tokens_seq_large_custom_tokenization_no_batch_test_epoch_7_predictions.tsv";
        let span_predictions_file_path = "/home/harsh/every-single-baseline/submission/old_predictions_test/experiment_multiconer_with_test_gold_labels_old_multiconer_fine_tokens_span_large_custom_tokenization_no_batch_test_epoch_7_predictions.tsv";
        let samples_file_path = "/home/harsh/every-single-baseline/preprocessed_data/multiconer_fine_test_tokens_production_samples.json";
        let seq_tsv_out = "./old_seq.tsv";
        let span_tsv_out = "./old_span.tsv";
        println!("TEST Seq");
        find_sub_token_predictions(seq_predictions_file_path, samples_file_path, seq_tsv_out);
        println!("TEST Span");
        find_sub_token_predictions(span_predictions_file_path, samples_file_path, span_tsv_out);
    }

    #[test]
    fn create_a_tsv_file() -> Result<(), Box<dyn Error>> {
        println!("Hello!");
        let mut writer = WriterBuilder::new().delimiter(b'\t').from_path("./test.tsv")?;
        writer.write_record(&["sample_id", "begin", "end", "type", "extraction"])?;
        Ok(())
    }


    #[test]
    fn write_aligned_predictions_test() {
        let span_predictions_file_path = "/home/harsh/every-single-baseline/submission/no_special_tokens_predictions/experiment_multiconer_with_test_gold_labels_multiconer_fine_vanilla_span_large_default_test_epoch_7_predictions.tsv";
        let seq_predictions_file_path = "/home/harsh/every-single-baseline/submission/no_special_tokens_predictions/experiment_multiconer_with_test_gold_labels_multiconer_fine_vanilla_model_seq_large_default_test_epoch_7_predictions.tsv";
        let samples_file_path = "/home/harsh/every-single-baseline/preprocessed_data/multiconer_fine_test_tokens_production_samples.json";
        let seq_tsv_out_path = "./seq_errors_aligned.tsv";
        let span_tsv_out_path = "./span_errors_aligned.tsv";
        println!("TEST Seq Aligned");
        write_aligned_predictions(seq_predictions_file_path, samples_file_path, seq_tsv_out_path);
        println!("TEST Span Aligned");
        write_aligned_predictions(span_predictions_file_path, samples_file_path, span_tsv_out_path);
    }

}
