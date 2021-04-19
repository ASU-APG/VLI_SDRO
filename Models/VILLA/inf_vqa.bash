python inf_vqa.py --txt_db /txt/vqa_devval_textattack.db --img_db /img/coco_val2014 \
    --output_dir /storage/villa_vqa/train_vqa_baseline_orig --checkpoint 4500 --pin_mem --fp16

python inf_vqa.py --txt_db /txt/vqa_test.db --img_db /img/coco_test2015 \
    --output_dir /storage/villa_vqa/train_vqa_baseline_orig --checkpoint 6000 --pin_mem --fp16


python inf_vqa.py --txt_db /txt/vqa_devval_si_sp.db --img_db /img/coco_val2014 \
    --output_dir /storage/villa_vqa/train_vqa_dataaug_20 --checkpoint 6000 --pin_mem --fp16

python inf_vqa.py --txt_db /txt/vqa_test.db --img_db /img/coco_test2015 \
    --output_dir /storage/villa_vqa/train_vqa_dataaug_20 --checkpoint 6000 --pin_mem --fp16


python inf_vqa.py --txt_db /txt/vqa_devval_si_sp.db --img_db /img/coco_val2014 \
    --output_dir /storage/villa_vqa/train_vqa_dataaug_40 --checkpoint 6000 --pin_mem --fp16

python inf_vqa.py --txt_db /txt/vqa_test.db --img_db /img/coco_test2015 \
    --output_dir /storage/villa_vqa/train_vqa_dataaug_40 --checkpoint 6000 --pin_mem --fp16



python inf_vqa.py --txt_db /txt/vqa_devval_textattack.db --img_db /img/coco_val2014 \
    --output_dir /storage/villa_vqa/train_vqa_groupdro_20 --checkpoint 5000 --pin_mem --fp16

python inf_vqa.py --txt_db /txt/vqa_test.db --img_db /img/coco_test2015 \
    --output_dir /storage/villa_vqa/train_vqa_groupdro_20 --checkpoint 6000 --pin_mem --fp16


python inf_vqa.py --txt_db /txt/vqa_devval_si_sp.db --img_db /img/coco_val2014 \
    --output_dir /storage/villa_vqa/train_vqa_groupdro_40 --checkpoint 6000 --pin_mem --fp16

python inf_vqa.py --txt_db /txt/vqa_test.db --img_db /img/coco_test2015 \
    --output_dir /storage/villa_vqa/train_vqa_groupdro_40 --checkpoint 6000 --pin_mem --fp16


python inf_vqa.py --txt_db /txt/vqa_devval_si_sp.db --img_db /img/coco_val2014 \
    --output_dir /storage/villa_vqa/train_vqa_sampledro_20 --checkpoint 5500 --pin_mem --fp16

python inf_vqa.py --txt_db /txt/vqa_test.db --img_db /img/coco_test2015 \
    --output_dir /storage/villa_vqa/train_vqa_sampledro_20 --checkpoint 6000 --pin_mem --fp16


python inf_vqa.py --txt_db /txt/vqa_devval_si_sp.db --img_db /img/coco_val2014 \
    --output_dir /storage/villa_vqa/train_vqa_sampledro_40 --checkpoint 4500 --pin_mem --fp16

python inf_vqa.py --txt_db /txt/vqa_test.db --img_db /img/coco_test2015 \
    --output_dir /storage/villa_vqa/train_vqa_sampledro_40 --checkpoint 6000 --pin_mem --fp16



python inf_vqa.py --txt_db /txt/vqa_devval_si_sp.db --img_db /img/coco_val2014 \
    --output_dir /storage/villa_vqa/train_vqa_groupdro_60_2 --checkpoint 1000 --pin_mem --fp16