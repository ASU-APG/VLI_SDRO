import torch
import json
import logging
logging.getLogger().setLevel(logging.INFO)
from fairseq.models.transformer import TransformerModel

en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
en2de.cuda()
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
de2en.cuda()

file1 = "/home/achaud39/Abhishek/Experiments/TVRetrieval/data/tvr_val_bckup.jsonl"
file2 = "/home/achaud39/Abhishek/Experiments/TVRetrieval/data/tvr_val_bckup_modified.jsonl"
output_file = open(file2, "w")
with open(file1, "r") as f:
    for idx, l in enumerate(f.readlines()):
        l = l.replace("\n", "")
        item = json.loads(l)
        sent = item['desc']
        paraphrase = de2en.translate(en2de.translate(sent))
        logging.info("{} ==== {} ==== {} ".format(idx, sent, paraphrase))
        item['desc'] = paraphrase
        st = json.dumps(item)
        output_file.write(st + "\n")

