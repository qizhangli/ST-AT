#!/bin/bash

python3 train_stat.py \
--dataset ${dataset} --epochs ${epochs} --beta ${beta} \
--model-dir results/${dataset}_stat_beta${beta}_${epochs}e 