#!/bin/bash

module load anaconda/2023a-pytorch
module load cuda/11.8

python3 train_vit.py --optim Adam --filename "out_adam.pt" --lr 1e-4 --epochs 1000
