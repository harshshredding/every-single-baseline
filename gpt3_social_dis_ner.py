import json
from preamble import *
from utils.easy_testing import get_valid_samples_by_dataset_name


def strip_and_lower_diseases(diseases: List[str]):
    return [disease.strip().lower() for disease in diseases]


def is_invalid_prediction(diseases: List[str]):
    cases_where_gpt_predicts_nothing = ['n/a', '-', 'none']
    if len(diseases) == 1:
        if diseases[0] in cases_where_gpt_predicts_nothing:
            return True
    return False


def find_cases_that_dont_match_exactly():
    with open('social_dis_ner_openai_output_valid.json', 'r') as gpt_predictions_file:
        gpt_predictions = json.load(gpt_predictions_file)

    # lowercase and trim
    gpt_predictions = [(sample_id, strip_and_lower_diseases(diseases)) for (sample_id, diseases) in gpt_predictions]
    # remove invalid predictions
    gpt_predictions = [(sample_id, diseases) for (sample_id, diseases) in gpt_predictions
                       if not is_invalid_prediction(diseases)]

    valid_samples = get_valid_samples_by_dataset_name('social_dis_ner')

    # make sure there is no sample that doesn't exist in the validation set
    valid_sample_ids = [sample.id for sample in valid_samples]
    for sample_id, diseases in gpt_predictions:
        assert sample_id in valid_sample_ids

    # lowercase all samples
    for sample in valid_samples:
        sample.text = sample.text.lower()

    valid_samples_dict = {sample.id: sample for sample in valid_samples}

    cases_not_substring = []

    for sample_id, gpt_diseases in gpt_predictions:
        sample = valid_samples_dict[sample_id]
        for predicted_disease in gpt_diseases:
            if predicted_disease not in sample.text:
                cases_not_substring.append((predicted_disease, sample.text))

    for case in cases_not_substring:
        print(case)


def main():
    with open('./social_dis_ner_openai_output_train.json', 'r') as gpt_predictions_file:
        gpt_predictions = json.load(gpt_predictions_file)
    print(len(gpt_predictions))
    gpt_predictions_dict = {
        sample_id: ','.join(diseases)
        for sample_id, diseases in gpt_predictions
    }
    print("dict len", len(gpt_predictions_dict))
    return gpt_predictions_dict


if __name__ == '__main__':
    main()
