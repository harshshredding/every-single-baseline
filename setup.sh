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
sudo apt-get install pkg-config
sudo apt-get -y install build-essential
pip install allennlp

echo "Setup project"
pip install -r requirements.txt
git config --global user.email "harshshredding@gmail.com"
git config --global user.name "Harsh Verma"
python download_data_files.py
python unzip_data_files.py