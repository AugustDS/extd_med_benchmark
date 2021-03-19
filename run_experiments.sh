#!/bin/bash -l 
# /train.py train data_dir result_dir random_seed resolution num_gpus learn_rate batch_size d_repeats total_kimg

# Brain CT scans
#python3 GAN_cpd/train.py train $2 $1/1.0/000 1000 256 8 0.001 256 1 4000
#python3 GAN_cpd/train.py train $2 $1/1.0/001 1000 256 8 0.003 256 1 4000
#python3 GAN_cpd/train.py train $2 $1/1.0/002 1000 256 8 0.005 256 1 4000
#python3 GAN_cpd/train.py train $2 $1/1.0/003 1000 256 8 0.008 256 1 4000
#python3 GAN_cpd/train.py train $2 $1/1.0/004 1000 256 8 0.01 256 1 4000
#python3 GAN_cpd/train.py train $2 $1/1.0/005 1000 256 8 0.015 256 1 4000
python3 GAN_cpd/train.py train $2 $1/1.0/007 1000 256 8 0.005 256 1 4000

#python3 GAN_cpd/train.py train $2 $1/2.0/000 1000 512 8 0.003 256 1 4000
#python3 GAN_cpd/train.py train $2 $1/2.0/001 1000 512 8 0.005 256 1 4000
#python3 GAN_cpd/train.py train $2 $1/2.0/002 1000 512 8 0.008 256 1 4000
#python3 GAN_cpd/train.py train $2 $1/2.0/003 1000 512 8 0.01 256 1 4000
#python3 GAN_cpd/train.py train $2 $1/2.0/005 1000 512 8 0.02 256 1 4000
python3 GAN_cpd/train.py train $2 $1/2.0/007 1000 512 8 0.005 256 1 4000
