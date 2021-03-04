#!/bin/bash -l 

# Brain Dataset
python3 /build_dataset/dataset_tool.py create_from_brain --tfrecord_dir /scratch/aschuette/brain_dataset/resolution --image_dir /scratch/aschuette/ct_scans_brain