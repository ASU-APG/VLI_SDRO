# SISP Transformations
Violin Annotations file can be downloaded from - https://drive.google.com/file/d/15XS7F_En90CHnSLrRmQ0M1bqEObuqt1-/view
All the file paths are provided in the config.yaml.

File Information
- get_noun_tokens.py is used to obtain nouns from violin dataset
- hyper_hypo.py is used to obtain the hyper-hyponyms for the nouns obtained by get_noun_tokens.py
- gen_cf.py has all the functions to generate counter factuals
- gen_sp.py has code for generation of sps

Both files gen_cf.py and gen_sp.py have dedicated functions to creates cfs and sps for violin and nlvr2. The filepaths are stored in config.yaml
- gen_sp requires setting up fairseq.
    - Clone the repository - https://github.com/pytorch/fairseq inside FairSeqNmt folder and follow the instructions to install.
- gen_sp also uses embedding files generated using - https://github.com/nesl/nlp_adversarial_examples (used in utils.py)


    
