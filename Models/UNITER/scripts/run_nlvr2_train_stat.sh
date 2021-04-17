CUDA_VISIBLE_DEVICES=2 nohup python train_nlvr2_stat.py --config config/train-nlvr2-base-1gpu-stat.json --output_dir /storage/uniter_nlvr2/nlvr2_all_data_stmt_groupstat_20 \
    --T 0.2 --iterative  > runoutputs/uniter_nlvr2_stat_group_20.out 2>runoutputs/uniter_nlvr2_stat_group_20.err &

#### SAMPLE ####

CUDA_VISIBLE_DEVICES=0 nohup python train_nlvr2_stat.py --config config/train-nlvr2-base-1gpu-stat.json --output_dir /storage/uniter_nlvr2/nlvr2_all_data_stmt_samplestat_20 \
    --T 0.2 --iterative --argmax_parents > runoutputs/uniter_nlvr2_stat_sample_20.out 2>runoutputs/uniter_nlvr2_stat_sample_20.err &
