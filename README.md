# Semantics Transformed Adversarial Training
## Introduction
Code for the ICCV 2020 paper ["Robust Vision-and-Language Inference via Semantics-Transformed AdversarialTraining"](). 

## SISP Transformations
Violin Annotations file can be downloaded from [here](https://drive.google.com/file/d/15XS7F_En90CHnSLrRmQ0M1bqEObuqt1-/view).

### Prerequisites
- SISP requires setting up [fairseq](https://github.com/pytorch/fairseq) (NMT).
- We also use embedding files generated using https://github.com/nesl/nlp_adversarial_examples

### Steps to generate SISP Transformations
- `get_noun_tokens.py` is used to obtain noun tokens for the dataset.
- `hyper_hypo.py` is used to obtain the hyper-hyponyms for the nouns obtained by get_noun_tokens.py
- `gen_si.py` is used to generate the Semantic Inverting Statements (SI) for a given sentence.
- `gen_sp.py` is used to generate the Semantic Preserving Statements (SP) for a given sentence.

## Models
Instructions to use the three models - [LXMERT](https://github.com/airsplay/lxmert), [UNITER](https://github.com/ChenRocks/UNITER), [VILLA](https://github.com/zhegan27/VILLA) can be found in their respective repository.
Additional `config` parameters are added in `train_nlvr2_stat.py` file like
- `n_pre` and `n_post` : Epoch range where STAT should be used.
- `T` : `%` of augmented samples to be used during training.
- `argmax_parents` : Set to true when using sample-wise STAT

A sample training command could be the following
```
python train_nlvr2_stat.py 
    --config config/train-nlvr2-base-1gpu-stat.json \
    --output_dir /storage/uniter_nlvr2 nlvr2_all_data_stmt_groupstat_20 \
    --T 0.2 \
    --argmax_parents
```




