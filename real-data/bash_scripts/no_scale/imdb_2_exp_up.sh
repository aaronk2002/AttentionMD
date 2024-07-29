#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

set CUDA_LAUNCH_BLOCKING = 1

python ../../train.py --config ../../configs/2/no_scale/imdb_config_2a.json
python ../../train.py --config ../../configs/2/no_scale/imdb_config_2b.json
python ../../train.py --config ../../configs/2/no_scale/imdb_config_2c.json