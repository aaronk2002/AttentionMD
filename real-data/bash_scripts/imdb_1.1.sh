#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

python ../train.py --config ../configs/1.1/imdb_config_1.1_masked_first.json
python ../train.py --config ../configs/1.1/imdb_config_1.1_masked_mean.json
python ../train.py --config ../configs/1.1/imdb_config_1.1_mean.json