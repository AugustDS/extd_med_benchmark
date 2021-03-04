# extd_med_benchmark

## Requirements 

- python3.7
- cuda/10.0
- cudnn/7.5-cu10.0

Install requirements.txt:

- `$ cd extd_med_benchmark`
- `$ pip install requirements.txt`

## Download raw dataset
Decide where to store raw data and set `\path_to_raw_data` accodringly 
- `$ bash download_dataset.sh \path_to_raw_data`

## Build tensorflow dataset
Decide where to store tf dataset and set `\path_to_tf_data` accodringly. This might require changes to the python3 command in build_dataset.sh to specify enough CPU memory for pre-processing the dataset. 
- `$ bash build_dataset.sh \path_to_raw_data \path_to_tf_data`

## Run GAN Training

- Each Job should be run on 8 GPUs with 16GB Memory (run on Pascal P100 nodes)
- Results will be saved under `\result_dir\1.0\000`,`\result_dir\1.0\001` etc. 
- Path to train.py file `\path_to_train_file` e.g. `\home\extd_med_benchmark\GAN_cpd\train.py`
- `$ bash run_experiments.sh \result_dir \path_to_tf_data \path_to_train_file`


## Once training is over 
- Share the `\result_dir\results.csv` file (Experiment + FID after 4M real images) 
