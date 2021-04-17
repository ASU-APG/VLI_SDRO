# group 20
CUDA_VISIBLE_DEVICES=0 nohup python train_vqa_stat.py \
    --config config/train-vqa-base-4gpu-stat.json \
    --T 0.2 \
    --iterative \
    --output_dir /storage/uniter_vqa/train_vqa_groupstat_20 > runoutputs/vqa/train_vqa_groupstat_20.out 2>runoutputs/vqa/train_vqa_groupstat_20.err &


# sample 20
CUDA_VISIBLE_DEVICES=2 nohup python train_vqa_stat.py \
    --config config/train-vqa-base-4gpu-stat.json \
    --T 0.2 \
    --iterative \
    --argmax_parents \
    --output_dir /storage/uniter_vqa/train_vqa_samplestat_20 > runoutputs/vqa/train_vqa_samplestat_20.out 2>runoutputs/vqa/train_vqa_samplestat_20.err &

