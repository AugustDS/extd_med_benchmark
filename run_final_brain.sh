#!/bin/bash -l 
# /train.py train data_dir result_dir load_network_run_id resume_kimgs random_seed resolution num_gpus learn_rate batch_size d_repeats total_kimg

# Brain CT scans
# 256^2: Restart from RUN 1.0/002 (FID 10.17)
python3 GAN_cpd/train.py train $2 $1/1.0/002 $1/1.0/002/network-final.pkl 4000 1000 256 8 0.005 160 1 12000
# 512^2: Restart from RUN 2.0/001 (FID 15.71)
python3 GAN_cpd/train.py train $2 $1/2.0/001 $1/2.0/001/network-final.pkl 4000 1000 512 8 0.005 64 1 12000
