#!/bin/bash

# Loading Modules in Supercloud
source /etc/profile
module load anaconda/2023b
module load cuda/11.8

# Variables
parameterization=W
att_svm_p=( 2 1_75 1_75 3 3 2 1_75 2 3 ) 
Ws_p=( 3 3 2 2 1_75 1_75 1_75 2 3 ) 
Ws_p_float=( 3 3 2 2 1.75 1.75 1.75 2 3 ) 

# For Loop
for seed in {1..100}
do
    for idx in "${!att_svm_p[@]}"
    do
        i=${att_svm_p[$idx]}
        j=${Ws_p[$idx]}
        p=${Ws_p_float[$idx]}
        python ../correlation_calculation.py --att_svm ../result/convergence/p$i/$parameterization$seed.pt --Ws ../result/convergence/p$j/$parameterization$seed.pt -p $p --output ../result/correlation/$i-$j/$parameterization$seed.pt
    done
done