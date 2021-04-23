# Semantics Transformed Adversarial Training
## Introduction
Code for the ICCV 2020 paper ["Robust Vision-and-Language Inference via Semantics-Transformed AdversarialTraining"](). 

## SISP Transformations
Violin Annotations file can be downloaded from [here](https://drive.google.com/file/d/15XS7F_En90CHnSLrRmQ0M1bqEObuqt1-/view).

### Contents
- `get_noun_tokens.py` is used to obtain noun tokens for the dataset.
- `hyper_hypo.py` is used to obtain the hyper-hyponyms for the nouns obtained by get_noun_tokens.py
- `gen_si.py` is used to generate the Semantic Inverting Statements (SI) for a given sentence.
- `gen_sp.py` is used to generate the Semantic Preserving Statements (SP) for a given sentence.

### Prerequisites
- SISP requires setting up [fairseq](https://github.com/pytorch/fairseq) (NMT).
- We also use embedding files generated using https://github.com/nesl/nlp_adversarial_examples

## Models
Instructions to use the three models - [LXMERT](https://github.com/airsplay/lxmert), [UNITER](https://github.com/ChenRocks/UNITER), [VILLA](https://github.com/zhegan27/VILLA) can be found in their respective repository.



