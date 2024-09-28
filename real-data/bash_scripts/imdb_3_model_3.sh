#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

python3 ../train.py --config ../configs/3/model_size_3.json