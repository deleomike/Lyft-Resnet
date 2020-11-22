#!/usr/bin

python -m venv env

source ./env/bin/activate

pip install l5kit==1.0.6 pytorch_pfn_extras

#TODO Prerequisites of horovod
#HOROVOD_GPU_OPERATIONS=NCCL pip install horovod

bash downloadData.sh

python -m main
