#!/bin/bash -l 
# /train.py train data_dir result_dir load_network_run_id resume_kimgs random_seed resolution num_gpus learn_rate batch_size d_repeats total_kimg compute_fid

# Chest X-Rays
# 256^2: Restart from RUN 3.0/002 (FID 31.07)
#python3 GAN_cpd/train.py train $2 $1/3.0/002 $1/3.0/002/network-final.pkl 4000 1000 256 8 0.005 160 1 12000
# 512^2: Restart from RUN 4.0/003 (FID 13.79)
#python3 GAN_cpd/train.py train $2 $1/4.0/003 $1/4.0/003/network-final.pkl 4000 1000 512 8 0.006 64 1 12000

#  Re-run 256^2 Chest X-Rays
#python3 GAN_cpd/train.py train $2 $1/8.0/000 None 0 2000 256 8 0.005 160 1 6257.0 0
#python3 GAN_cpd/train.py train $2 $1/8.0/001 None 0 3000 256 8 0.005 160 1 6257.0 0
#python3 GAN_cpd/train.py train $2 $1/8.0/002 None 0 4000 256 8 0.005 160 1 6257.0 0

#python3 GAN_cpd/train.py train $2 $1/8.0/003 None 0 5000 256 8 0.005 192 1 6257.0 0
#python3 GAN_cpd/train.py train $2 $1/8.0/004 None 0 6000 256 8 0.005 192 1 6257.0 0


# Re-run 512^2 Chest X-rays 6345.0
#python3 GAN_cpd/train.py train $2 $1/9.0/000 None 0 2000 512 8 0.006 64 1 6345.0 0
#python3 GAN_cpd/train.py train $2 $1/9.0/001 None 0 3000 512 8 0.006 64 1 6345.0 0
#python3 GAN_cpd/train.py train $2 $1/9.0/002 None 0 4000 512 8 0.006 64 1 6345.0 0
#python3 GAN_cpd/train.py train $2 $1/9.0/003 None 0 1000 512 8 0.006 96 1 6345.0 0
#python3 GAN_cpd/train.py train $2 $1/9.0/004 None 0 5000 512 8 0.006 96 1 6345.0 0

# Restart
#python3 GAN_cpd/train.py train $2 $1/9.0/000 $1/9.0/000 0 2000 512 8 0.006 64 1 6345.0 0
#python3 GAN_cpd/train.py train $2 $1/9.0/001 $1/9.0/001 0 3000 512 8 0.006 64 1 6345.0 0
#python3 GAN_cpd/train.py train $2 $1/9.0/002 $1/9.0/002 0 4000 512 8 0.006 64 1 6345.0 0
python3 GAN_cpd/train.py train $2 $1/9.0/004 $1/9.0/004 0 5000 512 8 0.006 96 1 6345.0 0

