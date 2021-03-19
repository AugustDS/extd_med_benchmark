#!/bin/bash -l 
# /train.py train data_dir result_dir random_seed resolution num_gpus learn_rate batch_size d_repeats total_kimg

python3 GAN_cpd/train.py train $2 $1/1.0/010 1000 256 8 0.005 256 1 4000
python3 GAN_cpd/train.py train $2 $1/2.0/010 1000 512 8 0.005 256 1 4000

