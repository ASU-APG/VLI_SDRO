# Dataaug 20
# CUDA_VISIBLE_DEVICES=1 nohup python train_vqa_adv.py \
#     --config config/train-vqa-base-4gpu-adv-aug-20.json \
#     --output_dir /storage/villa_vqa/train_vqa_dataaug_20 > runoutputs/vqa/train_vqa_dataaug_20.out 2>runoutputs/vqa/train_vqa_dataaug_20.err &

# Dataaug 60
CUDA_VISIBLE_DEVICES=0 nohup python train_vqa_adv.py \
    --config config/train-vqa-base-4gpu-adv-aug-60.json \
    --output_dir /storage/villa_vqa/train_vqa_dataaug_60_2 > runoutputs/vqa/train_vqa_dataaug_60_2.out 2>runoutputs/vqa/train_vqa_dataaug_60_2.err &

# Dataaug 80
CUDA_VISIBLE_DEVICES=1 nohup python train_vqa_adv.py \
    --config config/train-vqa-base-4gpu-adv-aug-80.json \
    --output_dir /storage/villa_vqa/train_vqa_dataaug_80_2 > runoutputs/vqa/train_vqa_dataaug_80_2.out 2>runoutputs/vqa/train_vqa_dataaug_80_2.err &

# Dataaug 100
CUDA_VISIBLE_DEVICES=2 nohup python train_vqa_adv.py \
    --config config/train-vqa-base-4gpu-adv-aug-100.json \
    --output_dir /storage/villa_vqa/train_vqa_dataaug_100_2 > runoutputs/vqa/train_vqa_dataaug_100_2.out 2>runoutputs/vqa/train_vqa_dataaug_100_2.err &




# CUDA_VISIBLE_DEVICES=1 nohup python train_vqa_adv.py \
#     --config config/train-vqa-base-4gpu-adv-aug-only-pos.json \
#     --output_dir /storage/villa_vqa/train_only_pos_vqa_dataaug_20 > runoutputs/vqa/train_only_pos_vqa_dataaug_20.out 2>runoutputs/vqa/train_only_pos_vqa_dataaug_20.err &    

# CUDA_VISIBLE_DEVICES=2 nohup python train_vqa_adv.py \
#     --config config/train-vqa-base-4gpu-adv-aug-only-si.json \
#     --output_dir /storage/villa_vqa/train_only_si_vqa_dataaug_20 > runoutputs/vqa/train_only_si_vqa_dataaug_20.out 2>runoutputs/vqa/train_only_si_vqa_dataaug_20.err &    

# CUDA_VISIBLE_DEVICES=3 nohup python train_vqa_adv.py \
#     --config config/train-vqa-base-4gpu-adv-aug-only-sp.json \
#     --output_dir /storage/villa_vqa/train_only_sp_vqa_dataaug_20 > runoutputs/vqa/train_only_sp_vqa_dataaug_20.out 2>runoutputs/vqa/train_only_sp_vqa_dataaug_20.err &    

# CUDA_VISIBLE_DEVICES=7 nohup python train_vqa_adv.py \
#     --config config/train-vqa-base-4gpu-adv-aug-80.json \
#     --output_dir /storage/villa_vqa/train_vqa_dataaug_80 > runoutputs/vqa/train_vqa_dataaug_80.out 2>runoutputs/vqa/train_vqa_dataaug_80.err &    