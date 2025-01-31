#!/bin/bash

# configuration
dataset='gaia'
labels_file='label_15_85.csv'
epochs=3000
lr=0.001
batch_size=128
guide_weight=0.1
aug_percent=0.2
seed=42

if [ "$dataset" = "gaia" ]; then
    python main.py --seed $seed --dataset $dataset --labels_file $labels_file  \
    --N_I 10 --N_T 5 --temperature 0.3  --epochs $epochs --lr $lr --batch_size $batch_size \
    --aggregator "lstm" --guide_weight $guide_weight --patience 5 --aug_percent $aug_percent \
    --dynamic_weight --TO --CM \
    --num_heads 16 --num_layers 2 --graph_hidden 128 --experiment_label "performance" \
    #--no_train --no_reconstruct \
    #--no_evaluate
fi