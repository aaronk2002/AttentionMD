#!/bin/bash

# Aggregated Result
for seed in {1..100}
do
   python ../local_convergence.py -n 6 -T 8 -d 10 -p 3 --lr 0.1 --epochs 20_000 --seed $seed --normalized --parameterization W --std 1 --output ../result/convergence/p3/W$seed.pt
done

# Single Result
python ../local_convergence.py -n 6 -T 8 -d 10 -p 3 --lr 0.1 --epochs 20_000 --seed 1 --normalized --parameterization W --std 0 --output ../result/convergence/p3/SingleW.pt
