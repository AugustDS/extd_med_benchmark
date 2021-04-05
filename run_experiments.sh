#!/bin/bash -l 
#-----------------------------------------
# HELP
#-----------------------------------------
# DAG IDENTIFIER: 0.dag: run (all), 1.dag: run (gan_test, cl_train_test), 2.dag: run (cl_train_test), 3.dag: run (cl_test)

#=========================================
# RUN TRAIN REAL TEST REAL
#=========================================
# bash bash_real_classifier.sh [current directory], [dataset directory] [experiment] [Random Seed] [dag_identifier] [resolution]
#-----------------------------------------
# CXPL
#bash bash_real_classifier_run.sh /home/aschuette/extd_med_benchmark /work/aschuette/brain_dataset_hr/resolution /0.2 /000 1000 6.dag 128

#Resolution Brain
bash bash_real_classifier_run.sh /home/aschuette/extd_med_benchmark /work/aschuette/brain_dataset_hr/resolution /0.3 /000 4000 2.dag 256
bash bash_real_classifier_run.sh /home/aschuette/extd_med_benchmark /work/aschuette/brain_dataset_hr/resolution /0.4 /000 4000 2.dag 512


#=========================================
# RUN EXPERIMENTS
#=========================================
# bash bash_run.sh 1-[current directory], 2-[dataset directory], 3-[result directory], 4-[experiment], 5-[trainings_run], 6-[Random Seed], 7-[dag_identifier], 8-[gan num gpu <=8], 9-[gan memory gpu <=500000] 10-[resolution]
#-----------------------------------------
# CXPL
#bash bash_run.sh /home/aschuette/extd_med_benchmark /work/aschuette/brain_dataset_hr/resolution /work/aschuette/extd_med_benchmark/results /2.2 /002 1000 6.dag 8 150000 128

# NNs
#bash bash_run.sh /home/aschuette/extd_med_benchmark /work/aschuette/brain_dataset_hr/resolution /work/aschuette/extd_med_benchmark/results /2.4 /000 1000 7.dag 8 150000 128

#Resolution
bash bash_run.sh /home/aschuette/extd_med_benchmark /work/aschuette/brain_dataset_hr/resolution /work/aschuette/extd_med_benchmark/results /1.0 /002 1000 1.dag 8 150000 256
bash bash_run.sh /home/aschuette/extd_med_benchmark /work/aschuette/brain_dataset_hr/resolution /work/aschuette/extd_med_benchmark/results /2.0 /001 1000 1.dag 8 250000 512

