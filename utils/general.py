from structs import Anno
import pandas as pd


def read_predictions_file(predictions_file_path) -> dict[str, list[Anno]]:
    df = pd.read_csv(predictions_file_path, sep='\t')
    sample_to_annos = {}
    for _, row in df.iterrows():
        annos_list = sample_to_annos.get(str(row['sample_id']), [])
        annos_list.append(
            Anno(
                begin_offset=int(row['begin']),
                end_offset=int(row['end']),
                label_type=str(row['type']),
                extraction=str(row['extraction']),
            )
        )
        sample_to_annos[str(row['sample_id'])] = annos_list
    return sample_to_annos
