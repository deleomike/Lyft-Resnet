#!/usr/bin

python -m venv env

source ./env/bin/activate

pip install l5kit pytorch_pfn_extras

bash downloadData.sh

python -m main
