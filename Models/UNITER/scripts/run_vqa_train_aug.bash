# Dataaug 20
CUDA_VISIBLE_DEVICES=0 nohup python train_vqa.py \
    --config config/train-vqa-base-4gpu-aug-20.json \
    --output_dir /storage/uniter_vqa/train_vqa_dataaug_20 > runoutputs/vqa/train_vqa_dataaug_20.out 2>runoutputs/vqa/train_vqa_dataaug_20.err &

# Dataaug 40
CUDA_VISIBLE_DEVICES=1 nohup python train_vqa.py \
    --config config/train-vqa-base-4gpu-aug-40.json \
    --output_dir /storage/uniter_vqa/train_vqa_dataaug_40 > runoutputs/vqa/train_vqa_dataaug_40.out 2>runoutputs/vqa/train_vqa_dataaug_40.err &    