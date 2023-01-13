#!/bin/sh

cd ~/

apt-get -y install vim

echo "Creating Anaconda environment"
conda init bash
conda create -n new_env python=3.10
. "/opt/conda/etc/profile.d/conda.sh"
conda activate new_env
echo "Done creating anaconda environment"

echo "Install Pytorch"
pip install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116

echo "Install Allennlp"
sudo apt-get install build-essential
pip install allennlp

echo "Setup project"
cd every-single-baseline/
pip install -r requirements.txt
git config --global user.email "harshshredding@gmail.com"
git config --global user.name "Harsh Verma"
python download_data_files.py
python unzip_data_files.py