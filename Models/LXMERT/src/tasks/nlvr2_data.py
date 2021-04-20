# coding=utf-8
# Copyleft 2019 project LXRT.

import json
import random
import numpy as np
from torch.utils.data import Dataset

from param import args
from utils import load_obj_tsv

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000


class NLVR2Dataset:
    """
    An NLVR2 data example in json file:
    {
        "identifier": "train-10171-0-0",
        "img0": "train-10171-0-img0",
        "img1": "train-10171-0-img1",
        "label": 0,
        "sent": "An image shows one leather pencil case, displayed open with writing implements tucked inside.
        ",
        "uid": "nlvr2_train_0"
    }/home/achaud39/Abhishek/Experiments/lxmert/data/nlvr2/train_inforground_modified.json
    """
    def __init__(self, splits: str):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        # print("Self.splits", splits)
        for split in self.splits:
            print("Split ", split)
            if split != 'test':
                all_data = json.load(open("/home/achaud39/Abhishek/Experiments/robustness_using_counterfactuals/files/nlvr/processed/%s_si_sp.json" % split))
            else: 
                all_data = json.load(open('text_attack2.json'))
        print(len(all_data))
        self.data.extend(all_data)
        #     aug_data, orig_data = [], []

        #     for item in all_data:
        #         # Only making use of positive stmts
        #         if item['tag'] != 'orig':
        #             aug_data.append(item)
        #         elif item['tag'] == 'orig': 
        #             orig_data.append(item)
                    
        #     del all_data
        #     orig_idx = len(orig_data)
        #     if args.test is None:
        #         random.shuffle(aug_data)
        #         random.shuffle(orig_data)
        #         aug_idx = int((args.adv_dataset * len(orig_data))/100)
        #         orig_idx = len(orig_data) - aug_idx
        #         print("Length of original data ", len(orig_data))
        #         print("Length of aug data ", aug_idx)
        #         self.data.extend(aug_data[:aug_idx])
        #     else: 
        #         self.data.extend(aug_data)

        #     self.data.extend(orig_data[:orig_idx])
            
        #     del orig_data
        #     del aug_data
        #     print("Total Data ", len(self.data))
        # all_data = json.load(open('contrast_set_nlvr2.json'))
        # print(len(all_data))
        # self.data.extend(all_data)
            

        print("Load %d data from split(s) %s." % (len(self.data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['uid']: datum
            for datum in self.data
        }

    def __len__(self):
        return len(self.data)


"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class NLVR2TorchDataset(Dataset):
    def __init__(self, dataset: NLVR2Dataset, ):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Loading detection features to img_data
        img_data = []
        if 'train' in dataset.splits:
            img_data.extend(load_obj_tsv('/scratch/achaud39/nlvr/nlvr2/data/nlvr2_imgfeat/train_obj36.tsv', topk=topk))
        if 'valid' in dataset.splits:
            img_data.extend(load_obj_tsv('/scratch/achaud39/nlvr/nlvr2/data/nlvr2_imgfeat/valid_obj36.tsv', topk=topk))
        if 'test' in dataset.name:
            img_data.extend(load_obj_tsv('/scratch/achaud39/nlvr/nlvr2/data/nlvr2_imgfeat/test_obj36.tsv', topk=topk))
        self.imgid2img = {}
        print("Len of image data", len(img_data))
        for img_datum in img_data:
            self.imgid2img[img_datum['img_id']] = img_datum

        # Filter out the dataset
        self.data = []
        for datum in self.raw_dataset.data:
            if datum['img0'] in self.imgid2img and datum['img1'] in self.imgid2img:
                self.data.append(datum)
        print("Use %d data in torch dataset" % (len(self.data)))
        print()

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item: int):
        datum = self.data[item]

        ques_id = datum['uid']
        ques = datum['sent']

        # Get image info
        boxes2 = []
        feats2 = []
        for key in ['img0', 'img1']:
            img_id = datum[key]
            img_info = self.imgid2img[img_id]
            boxes = img_info['boxes'].copy()
            feats = img_info['features'].copy()
            assert len(boxes) == len(feats)

            # Normalize the boxes (to 0 ~ 1)
            img_h, img_w = img_info['img_h'], img_info['img_w']
            boxes[..., (0, 2)] /= img_w
            boxes[..., (1, 3)] /= img_h
            np.testing.assert_array_less(boxes, 1+1e-5)
            np.testing.assert_array_less(-boxes, 0+1e-5)

            boxes2.append(boxes)
            feats2.append(feats)
        feats = np.stack(feats2)
        boxes = np.stack(boxes2)

        # Create target
        if 'label' in datum:
            label = datum['label']
            return ques_id, feats, boxes, ques, label
        else:
            return ques_id, feats, boxes, ques
    
    def get_aug_data():
        pass
        # one for iterative
        # one for t% random
        



class NLVR2Evaluator:
    def __init__(self, dataset: NLVR2Dataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans: dict):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans == label:
                score += 1
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans: dict, path):
        """
        Dump result to a CSV file, which is compatible with NLVR2 evaluation system.
        NLVR2 CSV file requirement:
            Each line contains: identifier, answer

        :param quesid2ans: nlvr2 uid to ans (either "True" or "False")
        :param path: The desired path of saved file.
        :return:
        """
        with open(path, 'w') as f:
            for uid, ans in quesid2ans.items():
                idt = self.dataset.id2datum[uid]["identifier"]
                ans = 'True' if ans == 1 else 'False'
                f.write("%s,%s\n" % (idt, ans))

