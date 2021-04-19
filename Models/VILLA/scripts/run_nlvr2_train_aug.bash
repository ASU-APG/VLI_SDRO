n_gpu=$1
T=$2
T2=$3
echo $n_gpu 
echo $T
echo $T2
CUDA_VISIBLE_DEVICES=$n_gpu \
    nohup python train_nlvr2_adv.py \
    --config config/train-nlvr2-base-1gpu-adv.json \
    --train_txt_db /txt/nlvr2_train_si_sp_${T}_orig_${T2}_aug.db \
    --output_dir /storage/nlvr2_train_si_sp_${T}_orig_${T2}_aug.db \
    --config config/train-nlvr2-base-1gpu-adv-aug.json > runoutputs/again/nlvr2_train_si_sp_${T}_orig_${T2}_aug.out 2> runoutputs/again/nlvr2_train_si_sp_${T}_orig_${T2}_aug.err &