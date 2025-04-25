#! /bin/bash
sudo apt-get install gcc make -y
sudo apt install build-essential linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh ./cuda_11.8.0_520.61.05_linux.run --silent --driver --toolkit
conda env create -n ia python=3.11.8
conda activate ia
pip install -r requirements.txt
# Driver 11.8 torch compatibilities
conda install pytorch==2.2.2 torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia