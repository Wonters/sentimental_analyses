#! /bin/bash
sudo apt-get install gcc make -y
sudo apt install build-essential linux-headers-$(uname -r)
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
sudo sh ./cuda_11.8.0_520.61.05_linux.run --silent --driver --toolkit