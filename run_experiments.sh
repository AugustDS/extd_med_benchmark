#!/bin/bash -l 
# /run_file.py data_dir result_dir random_seed resolution num_gpus learn_rate batch_size d_repeats total_kimg
python3 $3/train.py $2 $1/1.0/000 1000 256 8 0.005 256 1 20
python3 $3/train.py $2 $1/1.0/001 1000 256 8 0.005 256 1 20