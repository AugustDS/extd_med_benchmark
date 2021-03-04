#!/bin/bash -l 
# /train.py train data_dir result_dir random_seed resolution num_gpus learn_rate batch_size d_repeats total_kimg
python3 GAN_cpd/train.py train $2 $1/1.0/000 1000 256 8 0.001 160 1 4000
python3 GAN_cpd/train.py train $2 $1/1.0/001 1000 256 8 0.005 160 1 4000
python3 GAN_cpd/train.py train $2 $1/1.0/002 1000 256 8 0.01 160 1 4000
python3 GAN_cpd/train.py train $2 $1/1.0/003 1000 256 8 0.015 160 1 4000
python3 GAN_cpd/train.py train $2 $1/1.0/004 1000 256 8 0.02 160 1 4000
python3 GAN_cpd/train.py train $2 $1/1.0/005 1000 256 8 0.03 160 1 4000

python3 GAN_cpd/train.py train $2 $1/2.0/000 1000 512 8 0.001 64 1 4000
python3 GAN_cpd/train.py train $2 $1/2.0/000 1000 512 8 0.005 64 1 4000
python3 GAN_cpd/train.py train $2 $1/2.0/001 1000 512 8 0.01 64 1 4000
python3 GAN_cpd/train.py train $2 $1/2.0/002 1000 512 8 0.015 64 1 4000
python3 GAN_cpd/train.py train $2 $1/2.0/003 1000 512 8 0.02 64 1 4000
python3 GAN_cpd/train.py train $2 $1/2.0/005 1000 512 8 0.03 64 1 4000
