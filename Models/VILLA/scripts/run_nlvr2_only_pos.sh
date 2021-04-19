n_gpu=$1
T=$2
echo $n_gpu 
echo $T
CUDA_VISIBLE_DEVICES=$n_gpu \
    nohup python train_nlvr2_adv.py \
    --config config/train-nlvr2-base-1gpu-adv.json \
    --train_txt_db /txt/nlvr2_train_only_pos_20_aug.db \
    --output_dir /storage/nlvr2_train_only_pos_aug  > runoutputs/again/nlvr2_train_only_pos_aug.out 2> runoutputs/again/nlvr2_train_only_pos_aug.err &