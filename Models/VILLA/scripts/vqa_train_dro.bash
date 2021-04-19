# # group 20
# CUDA_VISIBLE_DEVICES=0 nohup python train_vqa_adv_dro.py \
#     --config config/train-vqa-base-4gpu-adv-dro.json \
#     --T 0.2 \
#     --output_dir /storage/villa_vqa/train_vqa_groupdro_20 > runoutputs/vqa/train_vqa_groupdro_20.out 2>runoutputs/vqa/train_vqa_groupdro_20.err &

# # group 40
# CUDA_VISIBLE_DEVICES=1 nohup python train_vqa_adv_dro.py \
#     --config config/train-vqa-base-4gpu-adv-dro.json \
#     --T 0.4 \
#     --output_dir /storage/villa_vqa/train_vqa_groupdro_40 > runoutputs/vqa/train_vqa_groupdro_40.out 2>runoutputs/vqa/train_vqa_groupdro_40.err &

# group 60
CUDA_VISIBLE_DEVICES=3 nohup python train_vqa_adv_dro.py \
    --config config/train-vqa-base-4gpu-adv-dro.json \
    --T 0.6 \
    --output_dir /storage/villa_vqa/train_vqa_groupdro_60_2 > runoutputs/vqa/train_vqa_groupdro_60_2.out 2>runoutputs/vqa/train_vqa_groupdro_60_2.err &

# group 80
CUDA_VISIBLE_DEVICES=4 nohup python train_vqa_adv_dro.py \
    --config config/train-vqa-base-4gpu-adv-dro.json \
    --T 0.7 \
    --output_dir /storage/villa_vqa/train_vqa_groupdro_80_2 > runoutputs/vqa/train_vqa_groupdro_80_2.out 2>runoutputs/vqa/train_vqa_groupdro_80_2.err &

# group 100
CUDA_VISIBLE_DEVICES=5 nohup python train_vqa_adv_dro.py \
    --config config/train-vqa-base-4gpu-adv-dro.json \
    --T 1.0 \
    --output_dir /storage/villa_vqa/train_vqa_groupdro_100_2 > runoutputs/vqa/train_vqa_groupdro_100_2.out 2>runoutputs/vqa/train_vqa_groupdro_100_2.err &

# # sample 20
# CUDA_VISIBLE_DEVICES=2 nohup python train_vqa_adv_dro.py \
#     --config config/train-vqa-base-4gpu-adv-dro.json \
#     --T 0.2 \
#     --argmax_parents \
#     --output_dir /storage/villa_vqa/train_vqa_sampledro_20 > runoutputs/vqa/train_vqa_sampledro_20.out 2>runoutputs/vqa/train_vqa_sampledro_20.err &

# # sample 40
# CUDA_VISIBLE_DEVICES=3 nohup python train_vqa_adv_dro.py \
#     --config config/train-vqa-base-4gpu-adv-dro.json \
#     --T 0.4 \
#     --argmax_parents \
#     --output_dir /storage/villa_vqa/train_vqa_sampledro_40 > runoutputs/vqa/train_vqa_sampledro_40.out 2>runoutputs/vqa/train_vqa_sampledro_40.err &

# sample 60
CUDA_VISIBLE_DEVICES=6 nohup python train_vqa_adv_dro.py \
    --config config/train-vqa-base-4gpu-adv-dro.json \
    --T 0.6 \
    --argmax_parents \
    --output_dir /storage/villa_vqa/train_vqa_sampledro_60_2 > runoutputs/vqa/train_vqa_sampledro_60_2.out 2>runoutputs/vqa/train_vqa_sampledro_60_2.err &

# sample 80
CUDA_VISIBLE_DEVICES=7 nohup python train_vqa_adv_dro.py \
    --config config/train-vqa-base-4gpu-adv-dro.json \
    --T 0.8 \
    --argmax_parents \
    --output_dir /storage/villa_vqa/train_vqa_sampledro_80_2 > runoutputs/vqa/train_vqa_sampledro_80_2.out 2>runoutputs/vqa/train_vqa_sampledro_80_2.err &

# sample 100
CUDA_VISIBLE_DEVICES=0 nohup python train_vqa_adv_dro.py \
    --config config/train-vqa-base-4gpu-adv-dro.json \
    --T 1.0 \
    --argmax_parents \
    --output_dir /storage/villa_vqa/train_vqa_sampledro_100_2 > runoutputs/vqa/train_vqa_sampledro_100_2.out 2>runoutputs/vqa/train_vqa_sampledro_100_2.err &

# # group 20 only pos
# CUDA_VISIBLE_DEVICES=4 nohup python train_vqa_adv_dro.py \
#     --config config/train-vqa-base-4gpu-adv-dro-only-pos.json \
#     --T 0.2 \
#     --output_dir /storage/villa_vqa/train_vqa_groupdro_20_only_pos > runoutputs/vqa/train_vqa_groupdro_20_only_pos.out 2>runoutputs/vqa/train_vqa_groupdro_20_only_pos.err &

# # group 20 only si
# CUDA_VISIBLE_DEVICES=5 nohup python train_vqa_adv_dro.py \
#     --config config/train-vqa-base-4gpu-adv-dro-only-si.json \
#     --T 0.2 \
#     --output_dir /storage/villa_vqa/train_vqa_groupdro_20_only_si > runoutputs/vqa/train_vqa_groupdro_20_only_si.out 2>runoutputs/vqa/train_vqa_groupdro_20_only_si.err &

# # group 20 only sp
# CUDA_VISIBLE_DEVICES=6 nohup python train_vqa_adv_dro.py \
#     --config config/train-vqa-base-4gpu-adv-dro-only-sp.json \
#     --T 0.2 \
#     --output_dir /storage/villa_vqa/train_vqa_groupdro_20_only_sp > runoutputs/vqa/train_vqa_groupdro_20_only_sp.out 2>runoutputs/vqa/train_vqa_groupdro_20_only_sp.err &


# # sample 20 only pos
# CUDA_VISIBLE_DEVICES=7 nohup python train_vqa_adv_dro.py \
#     --config config/train-vqa-base-4gpu-adv-dro-only-pos.json \
#     --T 0.2 \
#     --argmax_parents \
#     --output_dir /storage/villa_vqa/train_vqa_sampledro_20_only_pos > runoutputs/vqa/train_vqa_sampledro_20_only_pos.out 2>runoutputs/vqa/train_vqa_sampledro_20_only_pos.err &

# # sample 20 only si
# CUDA_VISIBLE_DEVICES=0 nohup python train_vqa_adv_dro.py \
#     --config config/train-vqa-base-4gpu-adv-dro-only-si.json \
#     --T 0.2 --argmax_parents \
#     --output_dir /storage/villa_vqa/train_vqa_sampledro_20_only_si > runoutputs/vqa/train_vqa_sampledro_20_only_si.out 2>runoutputs/vqa/train_vqa_sampledro_20_only_si.err &

# # sample 20 only sp
# CUDA_VISIBLE_DEVICES=1 nohup python train_vqa_adv_dro.py \
#     --config config/train-vqa-base-4gpu-adv-dro-only-sp.json \
#     --T 0.2 --argmax_parents \
#     --output_dir /storage/villa_vqa/train_vqa_sampledro_20_only_sp > runoutputs/vqa/train_vqa_sampledro_20_only_sp.out 2>runoutputs/vqa/train_vqa_sampledro_20_only_sp.err &