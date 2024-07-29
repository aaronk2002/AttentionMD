#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

set CUDA_LAUNCH_BLOCKING = 1

python ../../train.py --config ../../configs/1.1/no_scale/imdb_config_1.1c.json
python ../../train.py --config ../../configs/1.1/no_scale/imdb_config_1.1b.json
python ../../train.py --config ../../configs/1.1/no_scale/imdb_config_1.1a.json