"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

NLVR2 dataset
"""
import copy

import torch
import random
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
from cytoolz import concat

from .data import (DetectFeatTxtTokDataset, TxtTokLmdb, DetectFeatLmdb,
                   get_ids_and_lens, pad_tensors, get_ids_and_lens_stat, get_gather_index, get_lens_for_ids)
# from .data_stat import get_ids_and_lens_stat

from utils.logger import LOGGER

class Nlvr2PairedDataset_STAT(DetectFeatTxtTokDataset):
    def __init__(self, txt_db, img_db, use_img_type=True, orig_per = 0.1):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.aug_ids = []

        txt2img = txt_db.txt2img
        _lens, _orig_ids, self.orig_transformation_dict = get_ids_and_lens_stat(txt_db, orig_per)
        LOGGER.info("Len of orig_transformation_dict {}".format(len(self.orig_transformation_dict)))
        # LOGGER.info("Got the ids 1")

        # Filter the ids which have corresponding image feature file
        orig_lens, self.orig_ids = [], []
        for tl, id_ in zip(_lens, _orig_ids):
            flag = True
            for img in txt2img[id_]: 
                if img not in self.img_db.name2nbb:
                    flag = False
            if flag:
                orig_lens.append(tl)
                self.orig_ids.append(id_)

        self.train_lens = [2*tl + sum(self.img_db.name2nbb[img]
                                for img in txt2img[id_])
                     for tl, id_ in zip(orig_lens, self.orig_ids)]
        # LOGGER.info("Got the lens 2")
        self.use_img_type = use_img_type

        # Initially we train on original data
        self.train_ids = copy.deepcopy(self.orig_ids)
        print("Original training len", len(self.train_ids))

    def __len__(self):
        return len(self.train_ids)

    def _get_img_feat(self, fname):
        img_feat, bb = self.img_db[fname]
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        num_bb = img_feat.size(0)
        return img_feat, img_bb, num_bb

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        id_ = self.train_ids[i]
        example = self.txt_db[id_]

        target = example['target']
        outs = []
        for i, img in enumerate(example['img_fname']):
            img_feat, img_pos_feat, num_bb = self._get_img_feat(img)
            # To check for bias 
            # img_feat = img_feat * 0
            # print(torch.max(img_feat), torch.min(img_feat))
            # break
            ####
            # text input
            input_ids = copy.deepcopy(example['input_ids'])

            input_ids = [self.txt_db.cls_] + input_ids + [self.txt_db.sep]
            attn_masks = [1] * (len(input_ids) + num_bb)
            input_ids = torch.tensor(input_ids)
            attn_masks = torch.tensor(attn_masks)
            if self.use_img_type:
                img_type_ids = torch.tensor([i+1]*num_bb)
            else:
                img_type_ids = None

            outs.append((input_ids, img_feat, img_pos_feat,
                         attn_masks, img_type_ids))
        return tuple(outs), target

    def get_aug_data(self, use_iterative=True, n_pre=0, n_post=3, T = 0.2):
        print("Train len in get_aug_data", len(self.train_ids))
        random.shuffle(self.orig_ids)
        if use_iterative:
            self.x = int((len(self.orig_ids) * T)/(n_post - n_pre))
        else: 
            self.x = int(len(self.orig_ids) * T)
        
        return Nlvr2PairedDatasetEval_STAT(self.x, self.orig_ids, self.txt_db, self.img_db, self.orig_transformation_dict, self.use_img_type)

    
    def add_aug_data(self, aug_item_ids, use_iterative=True):
        # print(len(aug_item_ids), self.x)
        assert len(aug_item_ids) == self.x 

        random.shuffle(self.orig_ids)
        print("orig_ids Len", len(self.orig_ids))
        new_ids = []
        txt2img = self.txt_db.txt2img
        if use_iterative:
            self.aug_ids.extend(aug_item_ids)
            print("Aug_li len {}".format(len(self.aug_ids)))
            self.train_ids = []
            self.train_ids.extend(self.aug_ids)
            orig_idx = len(self.orig_ids) - len(self.aug_ids)
            print("Orig_li len {}".format(orig_idx))
            self.train_ids.extend(self.orig_ids[:orig_idx])
            _lens = get_lens_for_ids(self.txt_db, self.train_ids)
            self.train_lens = [2*tl + sum(self.img_db.name2nbb[img]
                                for img in txt2img[id_])
                            for tl, id_ in zip(_lens, self.train_ids)]
            print("Total train len", len(self.train_ids))
        else: 
            self.train_ids.extend(aug_item_ids)
            print("Aug_li len {}".format(len(self.aug_ids)))
            _lens = get_lens_for_ids(self.txt_db, self.train_ids)
            self.train_lens = [2*tl + sum(self.img_db.name2nbb[img]
                                for img in txt2img[id_])
                            for tl, id_ in zip(_lens, self.train_ids)]
            print("Total train len", len(self.train_ids))


    # def get_lens_for_ids(self, ids):
    #     return [2*tl + sum(self.img_db.name2nbb[img]
    #                             for img in txt2img[id_])
    #                  for tl, id_ in zip(txt_lens, ids)]



def nlvr2_paired_collate_stat(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks,
     img_type_ids) = map(list, unzip(concat(outs for outs, _ in inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    if img_type_ids[0] is None:
        img_type_ids = None
    else:
        img_type_ids = pad_sequence(img_type_ids,
                                    batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.Tensor([t for _, t in inputs]).long()

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'img_type_ids': img_type_ids,
             'targets': targets}
    return batch


class Nlvr2PairedDatasetEval_STAT(DetectFeatTxtTokDataset):

    def __init__(self, x, orig_ids, txt_db, img_db, orig_transformation_dict, use_img_type=True):
        assert isinstance(txt_db, TxtTokLmdb)
        assert isinstance(img_db, DetectFeatLmdb)
        self.txt_db = txt_db
        self.img_db = img_db
        self.orig_ids = orig_ids
        self.orig_transformation_dict = orig_transformation_dict
        self.x = x
        self.ids = []
        count = 0
        txt2img = self.txt_db.txt2img
        for parent_id in orig_ids:
            if parent_id not in self.orig_transformation_dict: continue 
            
            count += 1
            self.ids.extend(self.orig_transformation_dict[parent_id])
            if count >= x:
                break
        # LOGGER.info("Orig_ids {}".format(len(orig_ids)))
        # LOGGER.info("Count {}".format(count))
        # LOGGER.info("Len of ids in eval stat {}".format(len(self.ids)))    
        txt_lens = get_lens_for_ids(self.txt_db, self.ids)
        # LOGGER.info("Len of txt_lens in eval stat {}".format(len(txt_lens)))

        self.lens = [2*tl + sum(self.img_db.name2nbb[img]
                                for img in txt2img[id_])
                     for tl, id_ in zip(txt_lens, self.ids)]

        # LOGGER.info("Len of lens {}".format(len(self.lens)))
        self.use_img_type = use_img_type

    def __len__(self):
        return len(self.ids)

    def _get_img_feat(self, fname):
        img_feat, bb = self.img_db[fname]
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        num_bb = img_feat.size(0)
        return img_feat, img_bb, num_bb

    def __getitem__(self, i):
        """
        [[txt, img1],
         [txt, img2]]
        """
        id_ = self.ids[i]
        example = self.txt_db[id_]

        # if 'sp_tag' in example:
        #     tag = example['sp_tag']
        #     example['tag'] = example['sp_tag']
        # elif 'cf_tag' in example:
        #     tag = example['cf_tag']
        #     example['tag'] = example['cf_tag']
        # else: 
        #     tag = 'orig'
        #     example['tag'] = tag

        target = example['target']
        outs = []
        for i, img in enumerate(example['img_fname']):
            img_feat, img_pos_feat, num_bb = self._get_img_feat(img)

            # text input
            input_ids = copy.deepcopy(example['input_ids'])

            input_ids = [self.txt_db.cls_] + input_ids + [self.txt_db.sep]
            attn_masks = [1] * (len(input_ids) + num_bb)
            input_ids = torch.tensor(input_ids)
            attn_masks = torch.tensor(attn_masks)
            if self.use_img_type:
                img_type_ids = torch.tensor([i+1]*num_bb)
            else:
                img_type_ids = None

            outs.append((input_ids, img_feat, img_pos_feat,
                         attn_masks, img_type_ids))
        return tuple(outs), target, example['parent_identifier'], example['tag'], id_
    
    # def get_lens_for_ids(self, ids):
    #     return [2*tl + sum(self.img_db.name2nbb[img]
    #                             for img in txt2img[id_])
    #                  for tl, id_ in zip(txt_lens, ids)]

def nlvr2_paired_collate_eval_stat(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks,
     img_type_ids) = map(list, unzip(concat(outs for outs, _,_,_,_ in inputs)))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    # image batches
    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)
    if img_type_ids[0] is None:
        img_type_ids = None
    else:
        img_type_ids = pad_sequence(img_type_ids,
                                    batch_first=True, padding_value=0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.Tensor([t for _, t, _, _, _ in inputs]).long()
    parent_ids = [p for _, _, p, _, _ in inputs]
    tags = [ta for _, _, _, ta, _ in inputs]
    item_ids = [_id for _ ,_ ,_ ,_ , _id in inputs]

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'img_type_ids': img_type_ids,
             'targets': targets,
             'parent_ids' : parent_ids,
             'tags' : tags,
             'item_ids' : item_ids}
    return batch