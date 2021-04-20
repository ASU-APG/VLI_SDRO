# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import os
import pickle
import random 
import numpy as np
import torch
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

VQA_DATA_ROOT = '/scratch/achaud39/VQA/'
MSCOCO_IMGFEAT_ROOT =  '/scratch/achaud39/VQA/data/mscoco_imgfeat'
SPLIT2NAME = {
    'train': 'train2014',
    'train_yesno': 'train2014',
    'valid': 'val2014',
    'minival': 'val2014',
    'minival_yesno': 'val2014',
    'minival_yesno_si': 'val2014',
    'minival_yesno_sp': 'val2014',
    'minival_yesno_sisp': 'val2014',
    'nominival': 'val2014',
    'nominival_yesno': 'val2014',
    'test': 'test2015',
}


class VQADataset_DRO:
    """
    A VQA data example in json file:
        {
            "answer_type": "other",
            "img_id": "COCO_train2014_000000458752",
            "label": {
                "net": 1
            },
            "question_id": 458752000,
            "question_type": "what is this",
            "sent": "What is this photo taken looking through?"
        }
    """
    def __init__(self, splits, aug_item_ids = [], aug_eval = False):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets
        self.data, self.only_orig_data = [], []
        self.orig_transformation_dict = {}
        self.data = []
        print(len(aug_item_ids))
        all_aug_data, filtered_aug_data = [], []
        all_data = []
        for split in self.splits:
            # self.data.extend(json.load(open(VQA_DATA_ROOT + "%s.json" % split)))
            # Loading si file, it also has original data along with sis
            all_data.extend(json.load(open(VQA_DATA_ROOT + "%s_yesno_si.json" % split)))
            # Loading sp file
            all_data.extend(json.load(open(VQA_DATA_ROOT + "%s_yesno_sp.json" % split)))

            
        for item in all_data:
            
            if item['tag'] == 'orig':
                self.only_orig_data.append(item)
            
            else: 
                parent_id = item['parent_question_id']
                if parent_id not in self.orig_transformation_dict:
                    self.orig_transformation_dict[parent_id] = []
                self.orig_transformation_dict[parent_id].append(item)

                all_aug_data.append(item)

                if item['question_id'] in aug_item_ids:
                    filtered_aug_data.append(item)

        # filtered_aug_data = list(set(filtered_aug_data))
        print("Filtered ", len(filtered_aug_data), "Aug_item_ids", len(aug_item_ids))
        assert len(filtered_aug_data) >= len(aug_item_ids)

        # If we are evaluating on aug_data just use filtered_aug_data
        if aug_eval == True:
            print("Len of filtered aug data", len(filtered_aug_data))
            self.data.extend(filtered_aug_data)
        else: 
            # While training use part of aug and part of only orig
            if 'train' in splits:
                random.shuffle(self.only_orig_data)
                orig_idx = len(self.only_orig_data) - len(filtered_aug_data)
                print("Length of original data ", orig_idx)
                print("Length of aug data ", len(filtered_aug_data))

                self.data.extend(filtered_aug_data)
                self.data.extend(self.only_orig_data[:orig_idx])
                # self.data.extend(self.only_orig_data)
            else:
                # While test and validation use all
                self.data.extend(all_aug_data) 
                self.data.extend(self.only_orig_data)

        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # Convert list to dict (for evaluation)
        self.id2datum = {
            datum['question_id']: datum
            for datum in self.data
        }

        # Answers
        # Check this
        self.ans2label = json.load(open("data/vqa/trainval_ans2label.json"))
        self.label2ans = json.load(open("data/vqa/trainval_label2ans.json"))
        assert len(self.ans2label) == len(self.label2ans)

    @property
    def num_answers(self):
        return len(self.ans2label)

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class VQATorchDataset_DRO(Dataset):
    def __init__(self, dataset: VQADataset_DRO, image_data):
        super().__init__()
        self.raw_dataset = dataset

        # if args.tiny:
        #     topk = TINY_IMG_NUM
        # elif args.fast:
        #     topk = FAST_IMG_NUM
        # else:
        #     topk = None

        # Loading detection features to img_data
        img_data = []
        for split in dataset.splits:
            # Minival is 5K images in MS COCO, which is used in evaluating VQA/LXMERT-pre-training.
            # It is saved as the top 5K features in val2014_***.tsv
            # load_topk = 5000 if (split == 'minival' and topk is None) else topk
            # img_data.extend(load_obj_tsv(
            #     os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
            #     topk=load_topk,
            #     fp16=args.fp16))
            img_data.extend(image_data[split])

        # Convert img list to dict
        self.imgid2img = {}
        print("Len of image data", len(img_data))
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Only kept the data with loaded image features
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img_id'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        img_id = datum['img_id']
        ques_id = datum['question_id']
        ques = datum['sent']
        if 'parent_question_id' in datum:
            parent_uid = datum['parent_question_id']
        else: 
            parent_uid = -1
        tag = datum['tag']
        # Get image info
        img_info = self.imgid2img[img_id]
        obj_num = img_info['num_boxes']
        feats = img_info['features'].copy()
        boxes = img_info['boxes'].copy()
        assert obj_num == len(boxes) == len(feats)

        # Normalize the boxes (to 0 ~ 1)
        img_h, img_w = img_info['img_h'], img_info['img_w']
        boxes = boxes.copy()
        boxes[:, (0, 2)] /= img_w
        boxes[:, (1, 3)] /= img_h
        np.testing.assert_array_less(boxes, 1+1e-2)
        np.testing.assert_array_less(-boxes, 0+1e-2)

        # Provide label (target)
        if 'label' in datum:
            label = datum['label']
            target = torch.zeros(self.raw_dataset.num_answers)
            for ans, score in label.items():
                target[self.raw_dataset.ans2label[ans]] = score
            return ques_id, feats, boxes, ques, target, parent_uid, tag
        else:
            return ques_id, feats, boxes, ques

    def get_aug_data(self):
        random.shuffle(self.raw_dataset.only_orig_data)

        x = int((len(self.raw_dataset.only_orig_data) * args.T)/(args.N_post-args.N_pre))
        
        aug_item_ids = []
        count = 0
        for datum in self.raw_dataset.only_orig_data:
            parent_id = datum['question_id']
            # Not all samples have transformations
            if parent_id not in self.raw_dataset.orig_transformation_dict:
                continue

            if datum['img_id'] in self.imgid2img :
                count += 1    
                for aug_item in self.raw_dataset.orig_transformation_dict[parent_id]:
                    aug_item_ids.append(aug_item['question_id'])

                # if len(aug_item_ids) >= x:
                if count >= x:
                    break

            if count >= x:
                break
    
        return aug_item_ids, count


class VQAEvaluator:
    def __init__(self, dataset: VQADataset_DRO):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans in label:
                score += label[ans]
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump results to a json file, which could be submitted to the VQA online evaluation.
        VQA json file submission requirement:
            results = [result]
            result = {
                "question_id": int,
                "answer": str
            }

        :param quesid2ans: dict of quesid --> ans
        :param path: The desired path of saved file.
        """
        with open(path, 'w') as f:
            result = []
            for ques_id, ans in quesid2ans.items():
                result.append({
                    'question_id': ques_id,
                    'answer': ans
                })
            json.dump(result, f, indent=4, sort_keys=True)


