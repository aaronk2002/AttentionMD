#!/bin/bash

module load anaconda/2023a-pytorch
module load cuda/11.8

python3 train_vit.py --optim SMD --filename "out_md.pt" --lr 5e-2 --epochs 1000
