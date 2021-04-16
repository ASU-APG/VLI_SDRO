
import torch
from fairseq.models.transformer import TransformerModel

class MY_NMT:
    def __init__(self):
        self.en2de = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.en-ru.single_model', tokenizer='moses', bpe='fastbpe')
        self.en2de.cuda()
        self.de2en = torch.hub.load('pytorch/fairseq', 'transformer.wmt19.ru-en.single_model', tokenizer='moses', bpe='fastbpe')
        self.de2en.cuda()
    def translate(self, sent):
        return self.de2en.translate(self.en2de.translate(sent))