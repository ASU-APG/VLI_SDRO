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


class NLVR2Dataset_DRO:
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
    def __init__(self, splits):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data = []
        self.orig_data = []
        self.aug_data = []
        self.orig_transformation_dict = {}
        print("Self.splits", splits)
        for split in self.splits:
            print("Split ", split)
            all_data = json.load(open("/home/achaud39/Abhishek/Experiments/robustness_using_counterfactuals/files/nlvr/processed/%s_si_sp.json" % split))

            for item in all_data:
                if item['tag'] != 'orig' and item['orig_label'] == 1:
                    parent_id = item['parent_uid']
                    self.aug_data.append(item)
                    if parent_id not in self.orig_transformation_dict:
                        self.orig_transformation_dict[parent_id] = []
                    self.orig_transformation_dict[parent_id].append(item)
                elif item['tag'] == 'orig': 
                    self.orig_data.append(item)


            random.shuffle(self.orig_data)
            random.shuffle(self.aug_data)

            # count = 0
            # for item in self.orig_data:
            #     if item['uid'] not in self.orig_transformation_dict:
            #         # print("Not found ", item['uid'])
            #         count += 1
            # print(count)
            del all_data

        
            print("Total Original Data ", len(self.orig_data))
        print("Load %d data from split(s) %s." % (len(self.orig_data), self.name))

        # List to dict (for evaluation and others)
        self.id2datum = {
            datum['uid']: datum
            for datum in self.orig_data
        }
        
        for datum in self.aug_data:
            self.id2datum[datum['uid']] = datum

    def __len__(self):
        return len(self.orig_data)



"""
An example in obj36 tsv:
FIELDNAMES = ["img_id", "img_h", "img_w", "objects_id", "objects_conf",
              "attrs_id", "attrs_conf", "num_boxes", "boxes", "features"]
FIELDNAMES would be keys in the dict returned by load_obj_tsv.
"""
class NLVR2TorchDataset_DRO(Dataset):
    def __init__(self, dataset):
        super().__init__()
        self.raw_dataset = dataset

        if args.tiny:
            topk = TINY_IMG_NUM
        elif args.fast:
            topk = FAST_IMG_NUM
        else:
            topk = -1

        # Getting image data from the caller
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
        self.orig_data1 = [] # Data which is used for training orig + aug_data
        self.orig_data2 = [] # List with only original data used for sampling orig data
        self.aug_data = [] # List for aug_data
        for datum in self.raw_dataset.orig_data:
            if datum['img0'] in self.imgid2img and datum['img1'] in self.imgid2img:
                self.orig_data1.append(datum)
                self.orig_data2.append(datum)
        
        # for datum in self.raw_dataset.aug_data:
        #     if datum['img0'] in self.imgid2img and datum['img1'] in self.imgid2img:
        #         self.aug_data.append(datum)   

        print("Use %d original data in torch dataset" % (len(self.orig_data2)))
        # print("Use %d aug data in torch dataset" % (len(self.aug_data)))
        print()

    def __len__(self):
        return len(self.orig_data1)

    def __getitem__(self, item):
        datum = self.orig_data1[item]

        ques_id = datum['uid']
        ques = datum['sent']
        if 'parent_uid' in datum:
            parent_uid = datum['parent_uid']
        else: 
            parent_uid = 'no_parent'

        # if 'sp_tag' in datum:
        #     tag = datum['sp_tag']
        # elif 'cf_tag' in datum:
        #     tag = datum['cf_tag']
        # else: 
        #     tag = 'orig'

        tag = datum['tag']

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
            return ques_id, feats, boxes, ques, label, parent_uid, tag
        else:
            return ques_id, feats, boxes, ques

    def get_eval_data(self):
        # Shuffle original data before getting the eval data
        random.shuffle(self.orig_data2)
        # self.x = int(self.orig_data_length * ((n-n_pre+1)*T)/(n_post-n_pre))
        if args.iterative:
            self.x = int((len(self.orig_data1) * args.T)/(args.n_post-args.n_pre))
        else:
            self.x = int(len(self.orig_data1) * args.T) 
        t = NLVR2Dataset_DRO_EVAL(self.x, self.imgid2img, self.raw_dataset, self.orig_data2)
        return t

    def add_aug_data(self, aug_data):
        print(len(aug_data), self.x)
        assert len(aug_data) == self.x
        random.shuffle(self.orig_data2)

        if args.iterative:
            self.aug_data.extend(aug_data)
            print("Aug_li len {}".format(len(self.aug_data)))
            self.orig_data1 = []
            self.orig_data1.extend(self.aug_data)

            orig_idx = len(self.orig_data2) - len(self.aug_data)
            print("Orig_li len {}".format(orig_idx))
            self.orig_data1.extend(self.orig_data2[:orig_idx]) 
        else: 
            self.orig_data1 = []
            print("Aug_li len {}".format(len(aug_data)))
            self.orig_data1.extend(aug_data)
            orig_idx = len(self.orig_data2) - len(aug_data)
            print("Orig_li len {}".format(orig_idx))
            self.orig_data1.extend(self.orig_data2[:orig_idx])

        assert len(self.orig_data1) == len(self.orig_data2)
        # assert  == self.orig_data_length

class NLVR2Dataset_DRO_EVAL(Dataset):
    def __init__(self, _x, _imgid2img, _raw_dataset, _orig_data):
        super().__init__()
        self.x = _x
        self.raw_dataset = _raw_dataset
        self.imgid2img = _imgid2img
        self.data = []
        self.orig_data = _orig_data

        orig_idx = _x
        print("Using {} aug_data".format(_x))
        count = 0
        parent_li = []
        for datum in self.orig_data:
            if datum['uid'] not in self.raw_dataset.orig_transformation_dict: continue
            # print("Parent",datum['uid'], len(self.raw_dataset.orig_transformation_dict[datum['uid']]))
            if datum['img0'] in self.imgid2img and datum['img1'] in self.imgid2img:
                count += 1
                self.data.extend(self.raw_dataset.orig_transformation_dict[datum['uid']])
                parent_li.append(datum['uid'])
            if count >= orig_idx:
                break
        assert count == orig_idx
        # print("Count", count, len(parent_li))
        print("Use %d eval data in torch dataset" % (len(self.data)))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        datum = self.data[item]

        ques_id = datum['uid']

        # if ques_id == 'nlvr2_train_4843':
        #     print(datum)

        ques = datum['sent']
        # if 'parent_id' in datum:
        parent_uid = datum['parent_uid']
        tag = datum['tag']
        # if 'sp_tag' in datum:
        #     tag = datum['sp_tag']
        #     datum['tag'] = datum['sp_tag']
        #     del datum['sp_tag']
        # elif 'cf_tag' in datum:
        #     tag = datum['cf_tag']
        #     datum['tag'] = datum['cf_tag']
        #     del datum['cf_tag']
        # else: 
        #     tag = 'orig'
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
            return ques_id, feats, boxes, ques, label, parent_uid, tag, datum
        else:
            return ques_id, feats, boxes, ques

    

class NLVR2Evaluator_DRO:
    def __init__(self, dataset):
        self.dataset = dataset

    def evaluate(self, quesid2ans):
        score = 0.
        for quesid, ans in quesid2ans.items():
            datum = self.dataset.id2datum[quesid]
            label = datum['label']
            if ans == label:
                score += 1
        return score / len(quesid2ans)

    def dump_result(self, quesid2ans, path):
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
