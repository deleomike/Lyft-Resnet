#!/bin/bash

#kaggle competitions download -c lyft-motion-prediction-autonomous-vehicles


#unzip lyft-motion-prediction-autonomous-vehicles.zip -d ./data

#kaggle datasets download -d lyft-full-training-set

mkdir data
cd data

#be in /data

mkdir scenes
cd scenes

#wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/train_full.tar
wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/sample.tar
#wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/train.tar
#wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/validate.tar

#tar -xvf train_full.tar
#rm train_full.tar

tar -xvf sample.tar
rm sample.tar

#tar -xvf train.tar
#rm train.tar

#tar -xvf validate.tar
#rm validate.tar

cd ..
mkdir aerial_map
cd aerial_map

wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/aerial_map.tar

tar -xvf aerial_map.tar
rm aerial_map.tar

cd ..
mkdir semantic_map
cd semantic_map

wget https://lyft-l5-datasets-public.s3-us-west-2.amazonaws.com/prediction/v1.1/semantic_map.tar

tar -xvf semantic_map.tar
rm semantic_map.tar


mv meta.json ..

cd ..
cd ..
