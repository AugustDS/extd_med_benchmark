#!/bin/bash -l 
# /train.py train data_dir result_dir load_network_run_id resume_kimgs random_seed resolution num_gpus learn_rate batch_size d_repeats total_kimg

# Brain CT scans
# 256^2: Restart from RUN 1.0/002 (FID 10.17)
#python3 GAN_cpd/train.py train $2 $1/1.0/002 $1/1.0/002/network-final.pkl 4000 1000 256 8 0.005 160 1 12000
# 512^2: Restart from RUN 2.0/001 (FID 15.71)
#python3 GAN_cpd/train.py train $2 $1/2.0/001 $1/2.0/001/network-final.pkl 4000 1000 512 8 0.005 64 1 12000

# 256^2 random seed repeats
#python3 GAN_cpd/train.py train $2 $1/5.0/000 None 0 2000 256 8 0.005 160 1 6096.0 0
#python3 GAN_cpd/train.py train $2 $1/5.0/001 None 0 3000 256 8 0.005 160 1 6096.0 0
#python3 GAN_cpd/train.py train $2 $1/5.0/002 None 0 4000 256 8 0.005 160 1 6096.0 0
python3 GAN_cpd/train.py train $2 $1/5.0/003 None 0 1000 256 8 0.005 192 1 6096.0 0

# 512^2 restart, test 2 LR again
#python3 GAN_cpd/train.py train $2 $1/6.0/000 None 0 1000 512 8 0.005 64 1 12000 1
#python3 GAN_cpd/train.py train $2 $1/6.0/001 None 0 1000 512 8 0.006 64 1 12000 1
python3 GAN_cpd/train.py train $2 $1/6.0/002 None 0 1000 512 8 0.005 64 1 12000 1
python3 GAN_cpd/train.py train $2 $1/6.0/003 None 0 1000 512 8 0.005 96 1 12000 1
python3 GAN_cpd/train.py train $2 $1/6.0/004 None 0 1000 512 8 0.005 96 1 12000 1

# Restart
# python3 GAN_cpd/train.py train $2 $1/6.0/000 $1/6.0/000 0 1000 512 8 0.005 64 1 12000 1
# python3 GAN_cpd/train.py train $2 $1/6.0/001 $1/6.0/001 0 1000 512 8 0.006 64 1 12000 1
