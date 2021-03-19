#!/bin/bash -l 
# /train.py train data_dir result_dir random_seed resolution num_gpus learn_rate batch_size d_repeats total_kimg

# Chest x-rays
python3 GAN_cpd/train.py train $2 $1/3.0/000 1000 256 8 0.003 256 1 4000
python3 GAN_cpd/train.py train $2 $1/3.0/001 1000 256 8 0.004 256 1 4000
python3 GAN_cpd/train.py train $2 $1/3.0/002 1000 256 8 0.005 256 1 4000
python3 GAN_cpd/train.py train $2 $1/3.0/002 1000 256 8 0.006 256 1 4000
python3 GAN_cpd/train.py train $2 $1/3.0/003 1000 256 8 0.007 256 1 4000

python3 GAN_cpd/train.py train $2 $1/4.0/000 1000 512 8 0.003 256 1 4000
python3 GAN_cpd/train.py train $2 $1/4.0/001 1000 512 8 0.004 256 1 4000
python3 GAN_cpd/train.py train $2 $1/4.0/002 1000 512 8 0.005 256 1 4000
python3 GAN_cpd/train.py train $2 $1/4.0/003 1000 512 8 0.006 256 1 4000
python3 GAN_cpd/train.py train $2 $1/4.0/005 1000 512 8 0.007 256 1 4000