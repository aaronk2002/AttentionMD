#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023b
module load cuda/11.8

# Single Result
python ../joint_training.py -p 3 --lr 0.1 --epochs 20_000 --output ../result/joint_convergence/3.pt