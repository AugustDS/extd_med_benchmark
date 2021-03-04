#!/bin/bash
module load cuda/10.0
module load cudnn/7.5-cu10.0
/work/aschuette/miniconda3/bin/python3 -Xfaulthandler /home/aschuette/extd_med_benchmark/GAN_cpd/train.py --data_dir /scratch/aschuette/brain_dataset/resolution --results_dir /work/aschuette/result_test_hr/1.0/000 --random_sed 1000 --resolution 256 --num_gpus 8 --learn_rate 0.005 --batch_size 256 --disc_repeats 1 --total_kimg 20
