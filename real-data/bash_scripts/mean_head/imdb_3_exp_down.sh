#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

set CUDA_LAUNCH_BLOCKING = 1

python ../../train.py --config ../../configs/3/mean_head/imdb_config_3e.json
python ../../train.py --config ../../configs/3/mean_head/imdb_config_3f.json
python ../../train.py --config ../../configs/3/mean_head/imdb_config_3g.json
python ../../train.py --config ../../configs/3/mean_head/imdb_config_3h.json