# Original
CUDA_VISIBLE_DEVICES=3 nohup python train_vqa.py \
    --config config/train-vqa-base-4gpu.json \
    --output_dir /storage/uniter_vqa/train_vqa_baseline_orig > runoutputs/vqa/train_vqa_baseline.out 2>runoutputs/vqa/train_vqa_baseline.err &    