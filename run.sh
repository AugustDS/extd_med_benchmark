#!/bin/bash
module load cuda/10.0
module load cudnn/7.5-cu10.0
/work/aschuette/miniconda3/bin/python3 -Xfaulthandler /home/aschuette/extd_med_benchmark/GAN_cpd/train.py train /scratch/aschuette/brain_dataset/resolution /work/aschuette/result_test_hr/1.0/000 1000 256 8 0.005 256 1 20
