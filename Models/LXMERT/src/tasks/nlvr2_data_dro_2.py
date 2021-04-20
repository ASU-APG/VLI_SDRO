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
TINY_IMG_NUM = 10000
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
    def __init__(self, splits, all_data, aug_item_ids = [], aug_eval = False):
        self.name = splits
        self.splits = splits.split(',')

        # Loading datasets to data
        self.data, self.only_orig_data = [], []
        self.orig_transformation_dict = {}
        # print("Self.splits", splits)
        for split in self.splits:
            # print("Split ", split)
            # all_data = json.load(open("/home/achaud39/Abhishek/Experiments/robustness_using_counterfactuals/files/nlvr/processed/%s_si_sp.json" % split))
            all_aug_data, filtered_aug_data = [], []

            for item in all_data:

                if item['tag'] == 'orig': 

                    self.only_orig_data.append(item)

                # elif item['orig_label'] == 1:
                else:

                    parent_id = item['parent_uid']
                    if parent_id not in self.orig_transformation_dict:
                        self.orig_transformation_dict[parent_id] = []
                    self.orig_transformation_dict[parent_id].append(item)

                    all_aug_data.append(item)
                    if item['uid'] in aug_item_ids:
                        filtered_aug_data.append(item)

            print("Filtered ", len(filtered_aug_data), "AUG_ITEM_IDS ", len(aug_item_ids))
            assert len(filtered_aug_data) == len(aug_item_ids)

            # If we are evaluating on aug_data just use filtered_aug_data
            if aug_eval == True:
                print("Len of filtered aug data", len(filtered_aug_data))
                self.data.extend(filtered_aug_data)
            
            elif aug_eval == False:
                # While training use part of aug and part of only orig
                if split == 'train':
                    random.shuffle(self.only_orig_data)
                    orig_idx = len(self.only_orig_data) - len(filtered_aug_data)
                    print("Length of original data ", orig_idx)
                    print("Length of aug data ", len(filtered_aug_data))

                    self.data.extend(filtered_aug_data)
                    self.data.extend(self.only_orig_data[:orig_idx])
                else:
                    # While test and validation use all
                    self.data.extend(all_aug_data) 
                    self.data.extend(self.only_orig_data)
            
            del all_data
            del filtered_aug_data
            del all_aug_data

        print("Total Data ", len(self.data))
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
    def __init__(self, dataset: NLVR2Dataset, img_data):
        super().__init__()
        self.raw_dataset = dataset

        # if args.tiny:
        #     topk = TINY_IMG_NUM
        # elif args.fast:
        #     topk = FAST_IMG_NUM
        # else:
        #     topk = -1

        # Loading detection features to img_data
        # img_data = []
        # if 'train' in dataset.splits:
        #     img_data.extend(load_obj_tsv('/scratch/achaud39/nlvr/nlvr2/data/nlvr2_imgfeat/train_obj36.tsv', topk=topk))
        # if 'valid' in dataset.splits:
        #     img_data.extend(load_obj_tsv('/scratch/achaud39/nlvr/nlvr2/data/nlvr2_imgfeat/valid_obj36.tsv', topk=topk))
        # if 'test' in dataset.name:
        #     img_data.extend(load_obj_tsv('/scratch/achaud39/nlvr/nlvr2/data/nlvr2_imgfeat/test_obj36.tsv', topk=topk))
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
        if 'parent_uid' in datum:
            parent_uid = datum['parent_uid']
        else: 
            parent_uid = 'no_parent'
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
    
    def get_aug_data(self):
        random.shuffle(self.raw_dataset.only_orig_data)

        x = int((len(self.raw_dataset.only_orig_data) * args.T)/(args.n_post-args.n_pre))
        
        aug_item_ids = []
        count = 0
        for datum in self.raw_dataset.only_orig_data:
            parent_id = datum['uid']
            # Not all samples have transformations
            if parent_id not in self.raw_dataset.orig_transformation_dict:
                continue

            if datum['img0'] in self.imgid2img and datum['img1'] in self.imgid2img:
                count += 1    
                for aug_item in self.raw_dataset.orig_transformation_dict[parent_id]:
                    aug_item_ids.append(aug_item['uid'])

                # if len(aug_item_ids) >= x:
                if count >= x
                    break

            if count >= x:
                break
    
        return aug_item_ids, count
        



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

