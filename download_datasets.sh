#!/bin/bash -l 

# Brain Dataset 18 GB
mkdir ct_scans_brain 
cd ct_scans_brain
kaggle datasets download -d backaggle/rsna_512
unzip rsna_512.zip
find ./stage_1_test_images_jpg -type f -name "*.jpg" -exec mv -f -t ./stage_1_train_images_jpg {} +
rm -r stage_1_test_images_jpg
mv stage_1_train_images_jpg stage_2_train_images
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=150QSmiXEua2E-xAGkP2f-VRGoHeLES3R' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=150QSmiXEua2E-xAGkP2f-VRGoHeLES3R" -O new_stage_2_train.csv && rm -rf /tmp/cookies.txt
rm -rf rsna_512.zip

# Chest X-Rays: 439 GB
#wget "https://us13.mailchimp.com/mctx/clicks?url=http%3A%2F%2Fdownload.cs.stanford.edu%2Fdeep%2FCheXpert-v1.0.zip&h=9784c0a4078a2e522334a8c3a6eb885721da9038bed2fd4eac39918e8a852689&v=1&xid=5747229e2e&uid=55365305&pool=contact_facing&subject=CheXpert-v1.0%3A+Subscription+Confirmed"
