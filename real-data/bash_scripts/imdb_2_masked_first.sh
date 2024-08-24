#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

python3 ../train.py --config ../configs/2/imdb_config_2_masked_first.json