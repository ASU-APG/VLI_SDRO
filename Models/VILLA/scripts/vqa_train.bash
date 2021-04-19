# Original
CUDA_VISIBLE_DEVICES=0 nohup python train_vqa_adv.py \
    --config config/train-vqa-base-4gpu-adv.json \
    --output_dir /storage/villa_vqa/train_vqa_baseline_orig > runoutputs/vqa/train_vqa_baseline.out 2>runoutputs/vqa/train_vqa_baseline.err &    