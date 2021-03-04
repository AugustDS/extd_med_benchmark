# extd_med_benchmark

## Requirements 

- python3.7
- cuda/10.0
- cudnn/7.5-cu10.0

Install requirements.txt:

- `$ cd extd_med_benchmark`
- `$ pip install requirements.txt`

## Download raw dataset
Decide where to store raw data and set `/path_to_raw_data` accodringly 
- `$ bash download_dataset.sh /path_to_raw_data`

## Build tensorflow dataset
Decide where to store tf dataset and set `/path_to_tf_data` accodringly. Building the dataset might require changes to the python3 command in build_dataset.sh to specify enough CPU memory for pre-processing the dataset. 
- Can be run on 1 CPU with 100000 MB storage request 
- `$ python3 build_dataset/dataset_tool.py create_from_brain --tfrecord_dir /path_to_tf_data --image_dir /path_to_raw_data`

## Run GAN training

- Each Job should be run on 8 GPUs with >16GB Memory (tested on 16GB Pascal P100 nodes)
- Results will be saved under `/result_dir/1.0/000`,`/result_dir/1.0/001` etc. 
- Might require changes to python3 commands in run_experiments.sh to specify GPU needs.
- `$ bash run_experiments.sh /result_dir /path_to_tf_data`


## Once training is over 
- Share the `/result_dir/results.csv` file (Experiment + FID after 4M real images) (and if possible some of the fake.png's in the result_subdir with the lowest FID score)
