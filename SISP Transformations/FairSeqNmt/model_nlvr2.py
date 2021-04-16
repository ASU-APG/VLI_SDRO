
import torch
import json
import logging
logging.getLogger().setLevel(logging.INFO)
from fairseq.models.transformer import TransformerModel
en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-de.single_model', tokenizer='moses', bpe='fastbpe')
# en2de.cuda()
de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.de-en.single_model', tokenizer='moses', bpe='fastbpe')
# de2en.cuda()

file1 = "/home/achaud39/Abhishek/Experiments/lxmert/data/nlvr2/test.json"
file2 = "/home/achaud39/Abhishek/Experiments/lxmert/data/nlvr2/test_para.json"

data = json.load(open(file1,'r'))
for idx, item in enumerate(data):
    sent = item['sent']
    item['sent'] = de2en.translate(en2de.translate(sent))
    logging.info("{} ==== {} ==== {} ".format(idx, sent, item['sent']))
    break
    #     stmts[i][1] = de2en.translate(en2de.translate(stmt_pair[1]))
    # clip['statement'] = stmts
    # entire_clip_info[item] = clip

# with open(file2, 'w') as fout:
#     json.dump(data , fout, sort_keys=True, indent=4)

