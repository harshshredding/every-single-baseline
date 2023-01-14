#!/bin/sh

apt-get -y install vim

echo "Creating Anaconda environment"
conda init bash
conda create -n new_env python=3.10
. "/opt/conda/etc/profile.d/conda.sh"
conda activate new_env
echo "Done creating anaconda environment"

echo "Installing Pytorch"
conda install pytorch torchvision torchaudio pytorch-cuda=11.6 -c pytorch -c nvidia

echo "Installing Allennlp"
sudo apt-get -y install build-essential
pip install allennlp

echo "Setup project"
pip install -r requirements.txt
git config --global user.email "harshshredding@gmail.com"
git config --global user.name "Harsh Verma"
python download_data_files.py
python unzip_data_files.py