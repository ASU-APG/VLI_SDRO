"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

VQA dataset
"""
import torch
from torch.nn.utils.rnn import pad_sequence
from toolz.sandbox import unzip
import json
import random
import copy
from os.path import abspath, dirname, exists, join
from .data_stat import DetectFeatTxtTokDataset, pad_tensors, get_gather_index, get_lens_for_ids, get_ids_and_lens_stat_vqa
from utils.logger import LOGGER

ans2label = json.load(open('/src/utils/ans2label.json'))

def _get_vqa_target(example, num_answers):

    target = torch.zeros(num_answers)
    temp_labels = example['target']['labels']

    labels = []
    for label in temp_labels: 
        labels.append(ans2label[label])

    scores = example['target']['scores']
    # print(labels, scores)
    if labels and scores:
        target.scatter_(0, torch.tensor(labels), torch.tensor(scores).float())
    return target

def _get_vqa_target_eval(example, num_answers):
    target = torch.zeros(num_answers)
    labels = example['target']['labels']
    scores = example['target']['scores']
    if labels and scores:
        target.scatter_(0, torch.tensor(labels), torch.tensor(scores))
    return target


class VqaDataset_DRO(DetectFeatTxtTokDataset):
    def __init__(self, num_answers, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Filter orig and aug data here
        self.txt_db = super().get_txt_db()
        self.img_db = super().get_img_db()
        txt2img = self.txt_db.txt2img
        self.aug_ids = []

        orig_lens, self.orig_ids, self.orig_transformation_dict = get_ids_and_lens_stat_vqa(self.txt_db)
        LOGGER.info("Len of orig_transformation_dict {}".format(len(self.orig_transformation_dict)))
        
        # Filter the ids which have corresponding image feature file
        # orig_lens, self.orig_ids = [], []
        # for tl, id_ in zip(_lens, _orig_ids):
        #     flag = True
        #     for img in [itxt2imgd_]: 
        #         if img not in self.img_db.name2nbb:
        #             print(img)
        #             flag = False
        #             break
        #     if flag:
        #         orig_lens.append(tl)
        #         self.orig_ids.append(id_)
        #     else :
        #         # print(id_)
        #         pass

        self.lens = [tl + self.img_db.name2nbb[txt2img[id_]]
                     for tl, id_ in zip(orig_lens, self.orig_ids)]


        # Initially we train on original data
        LOGGER.info("Len of orig_ids {}".format(len(self.orig_ids)))

        self.train_ids = copy.deepcopy(self.orig_ids)

        self.num_answers = num_answers

    def __len__(self):
        return len(self.train_ids)

    def _get_img_feat(self, fname):
        img_feat, bb = self.img_db[fname]
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        num_bb = img_feat.size(0)
        return img_feat, img_bb, num_bb

    def __getitem__(self, i):
        # example = super().__getitem__(i)
        id_ = self.train_ids[i]
        example = self.txt_db[id_]
        # print(example)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        target = _get_vqa_target(example, self.num_answers)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, target

    def get_aug_data(self, use_iterative=True, n_pre=0, n_post=3, T = 0.2):
        random.shuffle(self.orig_ids)
        if use_iterative:
            self.x = int((len(self.orig_ids) * T)/(n_post - n_pre))
        else: 
            self.x = int(len(self.orig_ids) * T)
        
        return VqaAugDataset_DRO(self.x, self.orig_ids, self.txt_db, self.img_db, self.orig_transformation_dict, self.num_answers), self.x


    def add_aug_data(self, aug_item_ids, use_iterative=True):
        print(len(aug_item_ids), self.x)
        # assert len(aug_item_ids) == self.x 

        random.shuffle(self.orig_ids)
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
            self.lens = [tl + self.img_db.name2nbb[txt2img[id_]]
                     for tl, id_ in zip(_lens, self.train_ids)]

        else: 
            pass


def vqa_collate_stat(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets}
    return batch


class VqaAugDataset_DRO(DetectFeatTxtTokDataset):
    def __init__(self, x, orig_ids, txt_db, img_db, orig_transformation_dict, num_answers):
        # super().__init__(*args, **kwargs)
        self.num_answers = num_answers
        self.x = x
        self.orig_ids = orig_ids
        self.txt_db = txt_db
        self.img_db = img_db
        self.orig_transformation_dict = orig_transformation_dict
        self.ids = []
        txt2img = self.txt_db.txt2img
        count = 0
        # print("Len of orig_ids ", len(orig_ids))
        for parent_id in orig_ids:
            if parent_id not in self.orig_transformation_dict: continue 
            
            count += 1
            self.ids.extend(self.orig_transformation_dict[parent_id])
            if count >= x:
                break

        # print("Count ", count)
        txt_lens = get_lens_for_ids(self.txt_db, self.ids)


        self.lens = [tl + self.img_db.name2nbb[txt2img[id_]]
                     for tl, id_ in zip(txt_lens, self.ids)]

    def __len__(self):
        return len(self.ids)

    def _get_img_feat(self, fname):
        img_feat, bb = self.img_db[fname]
        img_bb = torch.cat([bb, bb[:, 4:5]*bb[:, 5:]], dim=-1)
        num_bb = img_feat.size(0)
        return img_feat, img_bb, num_bb

    def __getitem__(self, i):
        # example = super().__getitem__(i)
        id_ = self.ids[i]
        example = self.txt_db[id_]
        # print(example)
        img_feat, img_pos_feat, num_bb = self._get_img_feat(
            example['img_fname'])

        # text input
        input_ids = example['input_ids']
        input_ids = self.txt_db.combine_inputs(input_ids)

        target = _get_vqa_target(example, self.num_answers)

        attn_masks = torch.ones(len(input_ids) + num_bb, dtype=torch.long)

        return input_ids, img_feat, img_pos_feat, attn_masks, target, example['tag'], example['parent_question_id'], id_


def vqa_collate_aug_stat(inputs):
    (input_ids, img_feats, img_pos_feats, attn_masks, targets, tags, parents, question_ids
     ) = map(list, unzip(inputs))

    txt_lens = [i.size(0) for i in input_ids]
    input_ids = pad_sequence(input_ids, batch_first=True, padding_value=0)
    position_ids = torch.arange(0, input_ids.size(1), dtype=torch.long
                                ).unsqueeze(0)

    attn_masks = pad_sequence(attn_masks, batch_first=True, padding_value=0)
    targets = torch.stack(targets, dim=0)

    num_bbs = [f.size(0) for f in img_feats]
    img_feat = pad_tensors(img_feats, num_bbs)
    img_pos_feat = pad_tensors(img_pos_feats, num_bbs)

    bs, max_tl = input_ids.size()
    out_size = attn_masks.size(1)
    gather_index = get_gather_index(txt_lens, num_bbs, bs, max_tl, out_size)

    batch = {'input_ids': input_ids,
             'position_ids': position_ids,
             'img_feat': img_feat,
             'img_pos_feat': img_pos_feat,
             'attn_masks': attn_masks,
             'gather_index': gather_index,
             'targets': targets,
             'tags' : tags,
             'parents' : parents,
             'question_ids' : question_ids}
    return batch
