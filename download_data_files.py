import gdown
import zipfile

url = "https://drive.google.com/file/d/1ifVNbPIlNfqRpPFiE3wofj6Vw03bwaA6/view?usp=sharing"
output = "gate_output.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/19rZltDngthsoEgY5DSwIaviWyX5xPV8s/view?usp=sharing"
output = "data_files.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/18bS-b1sXyiKV7wntyA7439UoxOIQeuJx/view?usp=sharing"
output = "embeddings.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1D3pIMdErfuI-H-ZJZOeVEElZJWKDgt60/view?usp=sharing"
output = "test_data.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)

url = "https://drive.google.com/file/d/1kWLlXosdCEcZ4kyd13qU2dcPrYBNLMDs/view?usp=sharing"
output = "gate-output-test.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)


url = "https://drive.google.com/file/d/1FFkl0eFXnnQZGQyARgtUPBzVJg0AbABk/view?usp=sharing"
output = "gate-output-old.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)


url = "https://drive.google.com/file/d/1JFY94OfozrihfCvp1QdDL3HeOxY0zW3n/view?usp=sharing"
output = "gate-output-big-diz-gaz.zip"
gdown.download(url=url, output=output, quiet=False, fuzzy=True)