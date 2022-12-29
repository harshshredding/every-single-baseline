# import pandas as pd
# from args import *
# from structs import *
# from typing import Dict
# from typing import List


# def get_train_annos_dict() -> Dict[str, List[Anno]]:
#     if curr_dataset == Dataset.social_dis_ner:
#         df = pd.read_csv(args['train_annos_file_path'], sep='\t')
#         sample_to_annos = {}
#         for i, row in df.iterrows():
#             annos_list = sample_to_annos.get(str(row['tweets_id']), [])
#             annos_list.append(Anno(row['begin'], row['end'], 'Disease', row['extraction']))
#             sample_to_annos[str(row['tweets_id'])] = annos_list
#         return sample_to_annos
#     elif curr_dataset == Dataset.few_nerd:
#         df = pd.read_csv(args['train_annos_file_path'], sep='\t')
#         sample_to_annos = {}
#         for i, row in df.iterrows():
#             annos_list = sample_to_annos.get(str(row['sample_id']), [])
#             annos_list.append(Anno(row['begin'], row['end'], row['type'], row['extraction']))
#             sample_to_annos[str(row['sample_id'])] = annos_list
#         return sample_to_annos
#     elif curr_dataset == Dataset.genia:
#         df = pd.read_csv(args['train_annos_file_path'], sep='\t')
#         sample_to_annos = {}
#         for i, row in df.iterrows():
#             annos_list = sample_to_annos.get(str(row['sample_id']), [])
#             annos_list.append(Anno(row['begin'], row['end'], row['type'], row['extraction']))
#             sample_to_annos[str(row['sample_id'])] = annos_list
#         return sample_to_annos
#     elif curr_dataset == Dataset.multiconer:
#         df = pd.read_csv(args['train_annos_file_path'], sep='\t')
#         sample_to_annos = {}
#         for i, row in df.iterrows():
#             annos_list = sample_to_annos.get(str(row['sample_id']), [])
#             annos_list.append(Anno(row['begin'], row['end'], row['type'], row['extraction']))
#             sample_to_annos[str(row['sample_id'])] = annos_list
#         return sample_to_annos
#     else:
#         raise Exception(f"{args['dataset_name']} is not supported")


# def get_valid_annos_dict() -> Dict[str, List[Anno]]:
#     if curr_dataset == Dataset.social_dis_ner:
#         df = pd.read_csv(args['valid_annos_file_path'], sep='\t')
#         sample_to_annos = {}
#         for i, row in df.iterrows():
#             annos_list = sample_to_annos.get(str(row['tweets_id']), [])
#             annos_list.append(Anno(row['begin'], row['end'], 'Disease', row['extraction']))
#             sample_to_annos[str(row['tweets_id'])] = annos_list
#         return sample_to_annos
#     elif curr_dataset == Dataset.few_nerd:
#         df = pd.read_csv(args['valid_annos_file_path'], sep='\t')
#         sample_to_annos = {}
#         for i, row in df.iterrows():
#             annos_list = sample_to_annos.get(str(row['sample_id']), [])
#             annos_list.append(Anno(row['begin'], row['end'], row['type'], row['extraction']))
#             sample_to_annos[str(row['sample_id'])] = annos_list
#         return sample_to_annos
