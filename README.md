# Semantics Transformed Adversarial Training
## Introduction
Code for ["Semantically Distributed Robust Optimization \\for Vision-and-Language Inference"](). 

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
- `n_pre` and `n_post` : Epoch range where SDRO should be used.
- `T` : `%` of augmented samples to be used during training.
- `argmax_parents` : Set to true when using sample-wise SDRO

A sample training command could be the following
```
python train_nlvr2_stat.py 
    --config config/train-nlvr2-base-1gpu-stat.json \
    --output_dir /storage/nlvr2_gw_stat_20 \
    --T 0.2 \
    --argmax_parents
```
The code we have provided may have some local paths to data that you would need to change.

### Evaluation
The trained SDRO models for NLVR2 ca be accessed via this [link](https://drive.google.com/file/d/1r3HbVhtGzzYwYUMziU3k3F3PpCnXdEIV/view?usp=sharing). The zip directory has the following :-
- models directory which has trained SW-SDRO and GW-SDRO models for UNITER and VILLA
- txt_db directory which has both training and testing SISP db
- ann directory which contains the JSON files used for created DB files.
Samples evaluation command
```
    python inf_nlvr2.py --txt_db /txt/nlvr2_test_si_sp.db/ \
        --img_db /img/nlvr2_test/ \
        --train_dir /storage/nlvr2_sw_stat --ckpt 8000 \
        --output_dir /storage/nlvr2_sw_stat --fp16

```
The results of evalution are written to `results.csv` in `output_dir`. The individual SI, SP and original accuracies can be viewed using `evaluate_nlvr2_test_results.py` script by modifying the path of results file.




