#!/bin/sh
pip install -r requirements.txt
apt-get -y install vim
git config --global user.email "harshshredding@gmail.com"
git config --global user.name "Harsh Verma"
python download_data_files.py
python unzip_data_files.py
