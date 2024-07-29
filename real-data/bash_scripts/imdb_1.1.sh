#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

set CUDA_LAUNCH_BLOCKING = 1

python ../train.py --config ../configs/1.1/imdb_config_1.1_acc_bulk.json
python ../train.py --config ../configs/1.1/imdb_config_1.1_epoch_bulk.json