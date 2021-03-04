#!/bin/bash -l 
# Brain Dataset
python3 /build_dataset/dataset_tool.py create_from_brain --tfrecord_dir $2 --image_dir $1