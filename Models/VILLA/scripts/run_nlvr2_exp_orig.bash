n_gpu=$1
T=$2
echo $n_gpu 
echo $T
CUDA_VISIBLE_DEVICES=$n_gpu \
    nohup python train_nlvr2_adv.py \
    --config config/train-nlvr2-base-1gpu-adv.json \
    --train_txt_db /txt/nlvr2_train_si_sp_${T}_orig.db \
    --output_dir /storage/villa_base_finetuned_${T}_orig \
    --config config/train-nlvr2-base-1gpu-adv-aug.json > runoutputs/again/villa_nlvr2_orig_$T.out 2> runoutputs/again/villa_nlvr2_orig_$T.err &