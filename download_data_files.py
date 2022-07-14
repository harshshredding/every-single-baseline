import gdown
import zipfile

url = "https://drive.google.com/file/d/1s9pOKV-GxTGKCfjOe_xWQVLuLAH3AP_m/view?usp=sharing"
output = "gate_output.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/19rZltDngthsoEgY5DSwIaviWyX5xPV8s/view?usp=sharing"
output = "data_files.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/18bS-b1sXyiKV7wntyA7439UoxOIQeuJx/view?usp=sharing"
output = "embeddings.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)
