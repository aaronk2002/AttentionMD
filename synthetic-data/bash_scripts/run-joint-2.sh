#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023b
module load cuda/11.8

# Single Result
python ../joint_training.py -p 2 --lr 0.1 --epochs 2_000 --output ../result/joint_convergence/2.pt