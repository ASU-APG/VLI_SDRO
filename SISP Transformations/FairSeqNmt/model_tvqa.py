import torch
import json
import logging
logging.getLogger().setLevel(logging.INFO)
from fairseq.models.transformer import TransformerModel
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
en2de.cuda()
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
de2en.cuda()

file = "/scratch/achaud39/tvqa/data/tvqa_val_processed.json"
file2 = "/scratch/achaud39/tvqa/data/tvqa_val_processed_modified.json"
data = json.load(open(file, 'r'))
res = list()

for idx, item in enumerate(data):
    q = item['q']
    paraphrase = de2en.translate(en2de.translate(q))
    logging.info("{} ==== {} ==== {} ".format(idx, q, paraphrase))
    item['q'] = paraphrase

with open(file2, 'w') as fout:
    json.dump(data , fout, sort_keys=True, indent=4)