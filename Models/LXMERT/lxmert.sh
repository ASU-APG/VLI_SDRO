#!/bin/bash
 
#SBATCH -N 1  # number of nodes
#SBATCH -n 10  # number of "tasks" (cores)
#SBATCH -t 00-10:15:00   # time in d-hh:mm:ss

#SBATCH -p cidsegpu1
#SBATCH -q cidsegpu1_contrib_res

#SBATCH --gres=gpu:1
#SBATCH -o RuntimeOutputs/nlvr2_test_sampledro_20_textattack.%j.out # file to save job's STDOUT (%j = JobId)
#SBATCH -e RuntimeOutputs/nlvr2_test_sampledro_20_textattack.%j.err # file to save job's STDERR (%j = JobId)
#SBATCH --mail-type=ALL # Send an e-mail when a job starts, stops, or fails
#SBATCH --mail-user=achaud39@asu.edu # Mail-to address

# Always purge modules to ensure consistent environments
module purge    
# source activate fairseq
cd /home/achaud39/Abhishek/Experiments/lxmert
# Load required modules for job's environment
source bin/activate
# bash run/nlvr2_finetune.bash 0 nlvr2_lxr955_tiny --tiny
# bash run/nlvr2_finetune.bash 0 20 nlvr2_train_all_data_dataaug_20
# bash run/nlvr2_finetune_sampledro.bash nlvr2_train_all_data_sampledro_20_2 0.2
bash run/nlvr2_test.bash 0 nlvr2_train_all_data_sampledro_20 --load /scratch/achaud39/lxmert_data/snap/nlvr2/nlvr2_train_all_data_sampledro_20/BEST --test test --batchSize 256
# bash run/nlvr2_finetune_dro.bash 0 nlvr2_train_n_pre0_n_post3_20_sample_argmax
# snap/nlvr2/nlvr2_train_all_data_sampledro_20
# bash run/nlvr2_finetune_dro.bash 0 nlvr2_lxr955_tiny_dro --tiny
# bash run/vqa_finetune_dro.bash 0 nlvr2_lxr955_tiny_dro --tiny
# /scratch/achaud39/lxmert_data/snap/nlvr2/nlvr2_lxr955_finetuned/BEST.pth