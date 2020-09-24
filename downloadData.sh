#!/bin/bash

kaggle competitions download -c lyft-motion-prediction-autonomous-vehicles


unzip lyft-motion-prediction-autonomous-vehicles.zip -d ./data

kaggle datasets download -d lyft-full-training-set

