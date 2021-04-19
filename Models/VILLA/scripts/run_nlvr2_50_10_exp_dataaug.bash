n_gpu=$1
T=$2
echo $n_gpu 
echo $T
CUDA_VISIBLE_DEVICES=$n_gpu \
    nohup python train_nlvr2_adv.py \
    --train_txt_db /txt/nlvr2_train_si_sp_aug_50_10.db \
    --val_txt_db /txt/nlvr2_dev.db \
    --test_txt_db /txt/nlvr2_test1.db \
    --output_dir /storage/villa_nlvr2/nlvr2_dataaug_50_10_orig_data \
    --config config/train-nlvr2-base-1gpu-adv-aug.json > runoutputs/nlvr2_dataaug_50_10_orig_data.out 2> runoutputs/nlvr2_dataaug_50_10_orig_data.err &