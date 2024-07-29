#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023b
module load cuda/11.8

# Single Result
python ../joint_training.py -p 1.75 --lr 0.1 --epochs 1_500 --output ../result/joint_convergence/1_75.pt