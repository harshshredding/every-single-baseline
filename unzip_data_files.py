import zipfile

with zipfile.ZipFile('./gate_output.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

with zipfile.ZipFile('./embeddings.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

with zipfile.ZipFile('./data_files.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

with zipfile.ZipFile('./test_data.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

with zipfile.ZipFile('./gate-output-test.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

with zipfile.ZipFile('./gate-output-old.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

with zipfile.ZipFile('./gate-output-big-diz-gaz.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

with zipfile.ZipFile('./gate-output-no-custom-tokenization.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

with zipfile.ZipFile('./few-nerd-dataset.zip', 'r') as zip_ref:
    zip_ref.extractall('./')