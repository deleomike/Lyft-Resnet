#!/usr/bin

sudo add-apt-repository ppa:graphics-drivers/ppa
sudo apt-get update

sudo systemctl stop nvidia-persistenced

sudo apt remove -y --allow-change-held-packages libcuda1-384 nvidia-384

sudo apt-get purge nvidia-*

sudo apt install nvidia-440
sudo apt install libcuda1-440
sudo apt install nvidia-cuda-toolkit


sudo reboot

curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_10.1.105-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_10.1.105-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub

# sudo apt-get update
# sudo apt-get install cuda

sudo reboot

