#!/bin/bash

cd semisup-adv
python robust_self_training_stat.py \
--aux_data_filename ../data/ti_500K_pseudo_labeled.pickle \
--distance l_inf --epsilon 0.031 \
--model wrn-28-10 \
--model_dir ../results/RST_STAT_beta6 --data_dir ../data \
--beta 6