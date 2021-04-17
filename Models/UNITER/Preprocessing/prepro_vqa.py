"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

preprocess NLVR annotations into LMDB
"""
import argparse
import json
import random
import os
from os.path import exists

from cytoolz import curry
from tqdm import tqdm
from pytorch_pretrained_bert import BertTokenizer

from data.data import open_lmdb

@curry
def bert_tokenize(tokenizer, text):
    ids = []
    for word in text.strip().split():
        ws = tokenizer.tokenize(word)
        if not ws:
            # some special char
            continue
        ids.extend(tokenizer.convert_tokens_to_ids(ws))
    return ids

def process_vqa_aug(annotation_file, db, tokenizer, missing=None, T = 0.2):

    with open(annotation_file, 'r') as infile:
        data = json.load(infile)

    count_orig, count_sisp = 0, 0
    for example in tqdm(data, desc='processing NLVR2'):
        if example['tag'] != 'orig':
            count_sisp += 1
        else:
            count_orig += 1
       
    want_sisp = int(T * count_orig )
    want_orig = count_orig - want_sisp 

    id2len = {}
    txt2img = {}  # not sure if useful
    img2txts = {}
    count1 = 0
    count2 = 0
    for example in tqdm(data, desc='processing VQA_YESNO'):
        count1+=1
        if example['tag'] == 'orig':
            if random.random() > want_orig/count_orig:
                continue 
        elif "yes" in example['orig_label']:
            if random.random() > want_sisp/count_sisp:
                continue
        else: 
            continue
        count2 += 1
        id_ = str(example['question_id'])

        img_fname = example['img_id'] + '.npz'
        img_fname = img_fname.replace("COCO", "coco")
        if missing and (img_fname[0] in missing or img_fname[1] in missing):
            continue
        input_ids = tokenizer(example['sent'])

        example['target'] = dict()

        example['target']['labels'] = list(example['label'].keys())
        example['target']['scores'] = list(example['label'].values())

        txt2img[id_] = img_fname
        id2len[id_] = len(input_ids)
        if img_fname in img2txts:
            img2txts["img_fname"].append(id_)
        else:
            img2txts["img_fname"] = id_
        example['input_ids'] = input_ids
        example['img_fname'] = img_fname
        db[id_] = example
    return id2len, txt2img, img2txts

def main(opts):
    if not exists(opts.output):
        os.makedirs(opts.output)
    else:
        raise ValueError('Found existing DB. Please explicitly remove '
                         'for re-processing')
    meta = vars(opts)
    meta['tokenizer'] = opts.toker
    meta['bert'] = opts.toker
    toker = BertTokenizer.from_pretrained(
        opts.toker, do_lower_case='uncased' in opts.toker)
    tokenizer = bert_tokenize(toker)
    meta['UNK'] = toker.convert_tokens_to_ids(['[UNK]'])[0]
    meta['CLS'] = toker.convert_tokens_to_ids(['[CLS]'])[0]
    meta['SEP'] = toker.convert_tokens_to_ids(['[SEP]'])[0]
    meta['MASK'] = toker.convert_tokens_to_ids(['[MASK]'])[0]
    meta['v_range'] = (toker.convert_tokens_to_ids('!')[0],
                       len(toker.vocab))
    with open(f'{opts.output}/meta.json', 'w') as f:
        json.dump(vars(opts), f, indent=4)

    open_db = curry(open_lmdb, opts.output, readonly=False)
    with open_db() as db:

        missing_imgs = None
        id2lens, txt2img, img2txts = process_vqa_aug(opts.annotation, db, tokenizer, missing_imgs)
    with open(f'{opts.output}/id2len.json', 'w') as f:
        json.dump(id2lens, f)
    with open(f'{opts.output}/txt2img.json', 'w') as f:
        json.dump(txt2img, f)
    with open(f'{opts.output}/img2txts.json', 'w') as f:
        json.dump(img2txts, f)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--annotation', required=True,
                        help='annotation JSON')
    parser.add_argument('--missing_imgs',
                        help='some training image features are corrupted')
    parser.add_argument('--output', required=True,
                        help='output dir of DB')
    parser.add_argument('--toker', default='bert-base-cased',
                        help='which BERT tokenizer to used')
    args = parser.parse_args()
    main(args)
