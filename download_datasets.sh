#!/bin/bash -l 

# Brain Dataset 18 GB
mkdir $1 
cd $1
kaggle datasets download -d backaggle/rsna_512
unzip rsna_512.zip
find ./stage_1_test_images_jpg -type f -name "*.jpg" -exec mv -f -t ./stage_1_train_images_jpg {} +
rm -r stage_1_test_images_jpg
mv stage_1_train_images_jpg stage_2_train_images
cd stage_2_train_images
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=150QSmiXEua2E-xAGkP2f-VRGoHeLES3R' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=150QSmiXEua2E-xAGkP2f-VRGoHeLES3R" -O new_stage_2_train.csv && rm -rf /tmp/cookies.txt
rm -rf rsna_512.zip
