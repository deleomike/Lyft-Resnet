#!/usr/bin

conda update conda

conda create -n Lyft python==3.7

conda activate Lyft

conda install -y gxx_linux-64

pip install l5kit==1.0.6 pytorch_pfn_extras horovod

#TODO Prerequisites of horovod
#HOROVOD_GPU_OPERATIONS=NCCL pip install horovod

bash downloadData.sh

#horovodrun -np 4 python -m main
