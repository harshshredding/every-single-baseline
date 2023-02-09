#!/bin/sh

apt-get -y install vim

echo "Creating Anaconda environment"
conda init bash
conda create -n nlp python=3.10 -y
. "/opt/conda/etc/profile.d/conda.sh"
conda activate nlp
echo "Done creating anaconda environment"

echo "Installing Pytorch"
conda install -y pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

echo "Installing Allennlp"
sudo apt-get -y install pkg-config
sudo apt-get -y install build-essential
pip install allennlp

echo "Install requirements"
pip install spacy
pip install openai
pip install pudb
pip install colorama
pip install gatenlp
pip install pandas
pip install transformers
pip install allennlp
pip install flair
pip install dropbox
pip install benepar

git config --global user.email "harshshredding@gmail.com"
git config --global user.name "Harsh Verma"
git config credential.helper store
python download_preprocessed_data.py
python unzip_data_files.py