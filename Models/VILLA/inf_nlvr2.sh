
# ORIGINAL
echo "Original"
# /data_1/data/uniter/txt_db/test_textattack.db
CUDA_VISIBLE_DEVICES=0 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
    --img_db /img/nlvr2_test/ \
    --train_dir /storage/villa_base_finetuned_recent/ --ckpt 8000 \
    --output_dir /storage/villa_base_finetuned_recent_si_sp_results/ --fp16

# # DATAAUG
# echo "AUG 1"
CUDA_VISIBLE_DEVICES=0 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
    --img_db /img/nlvr2_test/ \
    --train_dir /storage/villa_base_finetuned_nlvr2_train_only_sp_20 --ckpt 8000 \
    --output_dir /storage/villa_base_finetuned_nlvr2_train_only_sp_20 --fp16


# echo "AUG 2"
CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
    --img_db /img/nlvr2_test/ \
    --train_dir /storage/villa_nlvr2/nlvr2_dataaug_all_data_20 --ckpt 8000 \
    --output_dir /storage/villa_nlvr2/nlvr2t_dataaug_20 --fp16
# echo "AUG 3"
# CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
#     --img_db /img/nlvr2_test/ \
#     --train_dir /storage/villa_nlvr2/nlvr2_dataaug_all_data_60 --ckpt 6500 \
#     --output_dir /storage/villa_nlvr2/nlvr2_dataaug_all_data_60 --fp16
# echo "AUG 4"
# CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
#     --img_db /img/nlvr2_test/ \
#     --train_dir /storage/villa_nlvr2/nlvr2_dataaug_all_data_80 --ckpt 6500 \
#     --output_dir /storage/villa_nlvr2/nlvr2_dataaug_all_data_80 --fp16
# echo "AUG 5"
# CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
#     --img_db /img/nlvr2_test/ \
#     --train_dir /storage/villa_nlvr2/nlvr2_dataaug_all_data_100 --ckpt 6500 \
#     --output_dir /storage/villa_nlvr2/nlvr2_dataaug_all_data_100 --fp16

# GROUP
# echo "Group 1"
CUDA_VISIBLE_DEVICES=0 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
    --img_db /img/nlvr2_test/ \
    --train_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_20 --ckpt 8000 \
    --output_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_20 --fp16

CUDA_VISIBLE_DEVICES=0 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
    --img_db /img/nlvr2_test/ \
    --train_dir /storage/villa_base_finetuned_nlvr2_train_only_si_20 --ckpt 8000 \
    --output_dir /storage/villa_base_finetuned_nlvr2_train_only_si_20 --fp16

CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
    --img_db /img/nlvr2_test/ \
    --train_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_only_si_0.2_exp --ckpt 7500 \
    --output_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_only_si_0.2_exp --fp16

CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
    --img_db /img/nlvr2_test/ \
    --train_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_only_sp_0.2_exp  --ckpt 6000 \
    --output_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_only_sp_0.2_exp  --fp16

CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
    --img_db /img/nlvr2_test/ \
    --train_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_only_si_0.2_exp  --ckpt 8000 \
    --output_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_only_si_0.2_exp  --fp16

CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
    --img_db /img/nlvr2_test/ \
    --train_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_only_sp_0.2_exp  --ckpt 6000 \
    --output_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_only_sp_0.2_exp  --fp16

# echo "Group 2"
CUDA_VISIBLE_DEVICES=0 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
    --img_db /img/nlvr2_test/ \
    --train_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_20 --ckpt 8000 \
    --output_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_20 --fp16

# echo "Group 3"
# CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
#     --img_db /img/nlvr2_test/ \
#     --train_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_60 --ckpt 7500 \
#     --output_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_60_contrast_set_results --fp16

# echo "Group 4"
# CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/contrast_set.db/ \
#     --img_db /img/nlvr2_test/ \
#     --train_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_80 --ckpt 8000 \
#     --output_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_80_contrast_set_results --fp16

# echo "Group 5"
# CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/contrast_set.db/ \
#     --img_db /img/nlvr2_test/ \
#     --train_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_100 --ckpt 8000 \
#     --output_dir /storage/villa_nlvr2/nlvr2_all_data_groupdro_100_contrast_set_results --fp16

# # SAMPLE
# echo "Sample 1"
# CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
#     --img_db /img/nlvr2_test/ \
#     --train_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_20 --ckpt 8000 \
#     --output_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_20 --fp16
# echo "Sample 2"
# CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/contrast_set.db/ \
#     --img_db /img/nlvr2_test/ \
#     --train_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_40 --ckpt 8000 \
#     --output_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_40_contrast_set_results --fp16
# echo "Sample 3"
# CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/contrast_set.db/ \
#     --img_db /img/nlvr2_test/ \
#     --train_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_60 --ckpt 8000 \
#     --output_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_60_contrast_set_results --fp16
# echo "Sample 4"
# CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/contrast_set.db/ \
#     --img_db /img/nlvr2_test/ \
#     --train_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_80 --ckpt 8000 \
#     --output_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_80_contrast_set_results --fp16
# echo "Sample 5"
# CUDA_VISIBLE_DEVICES=5 python inf_nlvr2.py --txt_db /txt/contrast_set.db/ \
#     --img_db /img/nlvr2_test/ \
#     --train_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_100 --ckpt 8000 \
#     --output_dir /storage/villa_nlvr2/nlvr2_all_data_sampledro_100_contrast_set_results --fp16


# # 50 EXP
# CUDA_VISIBLE_DEVICES=0 python inf_nlvr2.py --txt_db /txt/nlvr2_test1.db/ \
#     --img_db /img/nlvr2_test/ \
#     --train_dir /storage/villa_nlvr2/nlvr2_dataaug_50_10_orig_data/ --ckpt 6500 \
#     --output_dir /storage/villa_nlvr2/nlvr2_dataaug_50_10_orig_data/ --fp16

# CUDA_VISIBLE_DEVICES=0 python inf_nlvr2.py --txt_db /txt/nlvr2_test1.db/ \
#     --img_db /img/nlvr2_test/ \
#     --train_dir /storage/villa_base_finetuned_50_orig/ --ckpt 6500 \
#     --output_dir /storage/villa_base_finetuned_50_orig/ --fp16
