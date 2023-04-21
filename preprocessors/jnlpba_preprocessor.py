
def read_jnlpba_iob_file() -> list[list]:
    iob_file_path = '/Users/harshverma/every-single-baseline/Genia4ERtraining/Genia4ERtask1.iob2'
    all_sample_tokens = []
    sentence_count = 0
    with open(iob_file_path, 'r') as file:
        curr_sample_tokens = []
        for line in file:
            line = line.strip()
            if len(line):
                word, tag = line.split('\t')
                assert word is not None
                assert tag is not None
                assert tag in [
                        'B-DNA',
                        'I-DNA',
                        'B-protein',
                        'I-protein',
                        'B-cell_type',
                        'I-cell_type',
                        'B-cell_line',
                        'I-cell_line',
                        'B-RNA',
                        'I-RNA',
                        'O'
                        ], \
                        f"Unexpected tag {tag}"
                curr_sample_tokens.append((word, tag))
            else:
                sentence_count += 1
                all_sample_tokens.append(curr_sample_tokens)
                curr_sample_tokens = []
    if len(curr_sample_tokens):
        all_sample_tokens.append(curr_sample_tokens)
    print("num_sentences", sentence_count)
    return all_sample_tokens
