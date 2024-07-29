#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

set CUDA_LAUNCH_BLOCKING = 1

python ../../train.py --config ../../configs/2/mean_head/imdb_config_2d.json
python ../../train.py --config ../../configs/2/mean_head/imdb_config_2e.json
python ../../train.py --config ../../configs/2/mean_head/imdb_config_2f.json