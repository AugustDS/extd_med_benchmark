# extd_med_benchmark

## Requirements 

- python3.7
- cuda/10.0
- cudnn/7.5-cu10.0

Install requirements.txt:

- `$ cd extd_med_benchmark`
- `$ pip install requirements.txt`

## Download raw dataset

- `$ bash download_dataset.sh \path_to_raw_data`

## Build tensorflow dataset

- `$ bash build_dataset.sh \path_to_raw_data \path_to_tf_data`

## Run GAN Training

- Each Job should be run on 8 GPUs with 16GB Memory (run on Pascal P100 nodes)
- 
