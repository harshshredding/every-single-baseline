import zipfile
from util import create_directory_structure

# with zipfile.ZipFile('./gate_output.zip', 'r') as zip_ref:
#     zip_ref.extractall('./')

with zipfile.ZipFile('./embeddings.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

# with zipfile.ZipFile('./data_files.zip', 'r') as zip_ref:
#     zip_ref.extractall('./')

# with zipfile.ZipFile('./test_data.zip', 'r') as zip_ref:
#     zip_ref.extractall('./')

# with zipfile.ZipFile('./gate-output-test.zip', 'r') as zip_ref:
#     zip_ref.extractall('./datasets/')
#
# with zipfile.ZipFile('./gate-output-old.zip', 'r') as zip_ref:
#     zip_ref.extractall('./datasets/')
#
# with zipfile.ZipFile('./gate-output-big-diz-gaz.zip', 'r') as zip_ref:
#     zip_ref.extractall('./datasets/')
#
# with zipfile.ZipFile('./gate-output-no-custom-tokenization.zip', 'r') as zip_ref:
#     zip_ref.extractall('./datasets/')
#
# with zipfile.ZipFile('./few-nerd-dataset.zip', 'r') as zip_ref:
#     zip_ref.extractall('./datasets/')

# with zipfile.ZipFile('./few-nerd-dataset.zip', 'r') as zip_ref:
#     zip_ref.extractall('./datasets/')

# with zipfile.ZipFile('./social-dis-ner-dataset.zip', 'r') as zip_ref:
#     zip_ref.extractall('./datasets/')

# with zipfile.ZipFile('./multiconer.zip', 'r') as zip_ref:
#     folder_path = "./datasets/multiconer"
#     create_directory_structure(folder_path)
#     zip_ref.extractall(folder_path)

# with zipfile.ZipFile('./legaleval.zip', 'r') as zip_ref:
#     folder_path = "./datasets/legaleval"
#     create_directory_structure(folder_path)
#     zip_ref.extractall(folder_path)

with zipfile.ZipFile('preprocessed_data.zip', 'r') as zip_ref:
    zip_ref.extractall('./')