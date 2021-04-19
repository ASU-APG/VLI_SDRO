n_gpu=$1
T=$2
echo $n_gpu 
echo $T
CUDA_VISIBLE_DEVICES=$n_gpu \
    nohup python train_nlvr2_adv.py \
    --config config/train-nlvr2-base-1gpu-adv.json \
    --train_txt_db /txt/nlvr2_train_only_sp_20.db \
    --output_dir /storage/nlvr2_train_only_sp_aug \
    --config config/train-nlvr2-base-1gpu-adv-aug.json > runoutputs/again/nlvr2_train_only_sp_aug.out 2> runoutputs/again/nlvr2_train_only_sp_aug.err &