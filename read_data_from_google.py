from typing import List, Dict
from structs import SampleId
from preamble import *
from annotators import get_google_search_headings
import json
from multiprocessing import Process

total_num_samples = 247947


def __get_text(tokens: List[tuple]) -> str:
    return ' '.join([token_string for token_string, _ in tokens])


def __read_raw_data(raw_file_path: str) -> Dict[SampleId, List[tuple]]:
    with open(raw_file_path, 'r') as dev_file:
        samples_dict = {}
        curr_sample_id = None
        for line in list(dev_file.readlines()):
            line = line.strip()
            if len(line):
                if line.startswith('#'):
                    sample_info = line.split()
                    assert (len(sample_info) == 4) \
                           or (len(sample_info) == 3)  # test files don't have domain in info
                    sample_id = sample_info[2]
                    curr_sample_id = sample_id
                else:
                    assert curr_sample_id is not None
                    split_line = line.split("_ _")
                    assert len(split_line) == 2
                    token_string = split_line[0].strip()
                    token_label = split_line[1].strip()
                    if not len(token_label):
                        token_label = 'O'
                    tokens_list = samples_dict.get(curr_sample_id, [])
                    tokens_list.append((token_string, token_label))
                    samples_dict[curr_sample_id] = tokens_list
        return samples_dict


def store_google_data(process_id, chunk_start, chunk_end):
    print(f"Starting google process {process_id}")
    raw_file_path = "multiconer-data-raw/public_data/EN-English/en_test.conll"
    all_sample_texts = []
    sample_to_tokens = __read_raw_data(raw_file_path)
    for sample_id in sample_to_tokens:
        tokens = sample_to_tokens[sample_id]
        sample_text = __get_text(tokens)
        all_sample_texts.append((sample_id, sample_text))
    assert len(all_sample_texts) == total_num_samples
    all_sample_texts = all_sample_texts[chunk_start: chunk_end]
    if chunk_end != total_num_samples:
        assert (chunk_end - chunk_start) == 24794
    else:
        assert chunk_start == 247940 and chunk_end == total_num_samples
    data_to_store = []
    for i, (sample_id, google_query) in enumerate(all_sample_texts):
        if (i % 100) == 0:
            print(f"pid {process_id} done with {i}")
        google_search_results = get_google_search_headings(google_query)
        assert len(google_search_results), f"query : {google_query}"
        print(google_search_results)
        search_result_string = ".".join(google_search_results)
        final_sample = ".".join([google_query, search_result_string])
        data_to_store.append({"sample_id": sample_id, "sample_text": final_sample})
    output_json_file_path = f'./multiconer_test_google_data_raw_{process_id}.json'
    with open(output_json_file_path, 'w') as output_file:
        json.dump(data_to_store, output_file)
    print(f"Finished google process {process_id}: {output_json_file_path}")


processes = []

for pid in range(11):
    chunk = total_num_samples // 10
    if pid == 10:
        process = Process(target=store_google_data, args=(pid, pid * chunk, total_num_samples))
    else:
        process = Process(target=store_google_data, args=(pid, pid * chunk, (pid + 1)*chunk))
    process.start()
    processes.append(process)

for process in processes:
    process.join()
