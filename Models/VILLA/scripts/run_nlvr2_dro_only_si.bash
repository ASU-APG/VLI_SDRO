n_gpu=$1
T=$2
echo $n_gpu 
echo $T

CUDA_VISIBLE_DEVICES=$n_gpu \
    nohup python train_nlvr2_adv_dro.py --config config/train-nlvr2-base-1gpu-adv-dro.json \
    --output_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_only_si \
    --test_txt_db /txt/nlvr2_test1.db/ \
    --argmax_parents \
    --T $T > runoutputs/again/nlvr2_all_data_sampledro_only_si.out 2>runoutputs/again/nlvr2_all_data_sampledro_only_si.err &