CUDA_VISIBLE_DEVICES=0 nohup python train_nlvr2.py \
    --config config/nlvr2-train-dataaug-20.json > runoutputs/uniter_nlvr2_aug_20.out 2> runoutputs/uniter_nlvr2_aug_20.err &

CUDA_VISIBLE_DEVICES=0 nohup python train_nlvr2.py \
    --config config/nlvr2-train-dataaug-40.json > runoutputs/uniter_nlvr2_aug_40.out 2> runoutputs/uniter_nlvr2_aug_40.err &

CUDA_VISIBLE_DEVICES=1 nohup python train_nlvr2.py \
    --config config/nlvr2-train-dataaug-60.json > runoutputs/uniter_nlvr2_aug_60.out 2> runoutputs/uniter_nlvr2_aug_60.err &

CUDA_VISIBLE_DEVICES=1 nohup python train_nlvr2.py \
    --config config/nlvr2-train-dataaug-80.json > runoutputs/uniter_nlvr2_aug_80.out 2> runoutputs/uniter_nlvr2_aug_80.err &

CUDA_VISIBLE_DEVICES=2 nohup python train_nlvr2.py \
    --config config/nlvr2-train-dataaug-100.json > runoutputs/uniter_nlvr2_aug_100.out 2> runoutputs/uniter_nlvr2_aug_0.err &