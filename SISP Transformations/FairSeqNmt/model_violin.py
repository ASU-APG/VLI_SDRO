import torch
import json
import logging
logging.getLogger().setLevel(logging.INFO)
from fairseq.models.transformer import TransformerModel
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
en2de.cuda()
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
de2en.cuda()

file1 = "/home/achaud39/Abhishek/Experiments/feat/violin_annotation.json"
file2 = "/home/achaud39/Abhishek/Experiments/feat/violin_annotation_paraphrased.json"

entire_clip_info = json.load(open(file1,'r'))
for item in entire_clip_info:
    clip = entire_clip_info[item]
    stmts = clip['statement']
    for i in range(len(stmts)):
        stmt_pair = stmts[i]
        stmts[i][0] = de2en.translate(en2de.translate(stmt_pair[0]))
        logging.info("{} ==== {} ==== {} ".format(item, stmt_pair[0], stmts[i][0]))
        stmts[i][1] = de2en.translate(en2de.translate(stmt_pair[1]))
    clip['statement'] = stmts
    entire_clip_info[item] = clip

with open(file2, 'w') as fout:
    json.dump(entire_clip_info , fout, sort_keys=True, indent=4)
