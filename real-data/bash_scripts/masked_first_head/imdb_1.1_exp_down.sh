#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023a-pytorch
module load cuda/11.8

set CUDA_LAUNCH_BLOCKING = 1

python ../../train.py --config ../../configs/1.1/masked_first_head/imdb_config_1.1d.json
python ../../train.py --config ../../configs/1.1/masked_first_head/imdb_config_1.1e.json
python ../../train.py --config ../../configs/1.1/masked_first_head/imdb_config_1.1f.json