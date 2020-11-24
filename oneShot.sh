#!/usr/bin

sudo add-apt-repository ppa:deadsnakes/ppa
sudo apt-get update

sudo apt-get install Python3.7 python3.7-venv

sudo apt-get install -y python3-venv

python3 -m venv env

source ./env/bin/activate

pip install l5kit==1.0.6 pytorch_pfn_extras

#TODO Prerequisites of horovod
#HOROVOD_GPU_OPERATIONS=NCCL pip install horovod

bash downloadData.sh

#horovodrun -np 4 python -m main
