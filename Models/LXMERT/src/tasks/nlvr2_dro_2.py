# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import json
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from lxrt.optimization import BertAdam
from param import args
from tasks.nlvr2_model import NLVR2Model
from tasks.nlvr2_data_dro_2 import NLVR2Dataset, NLVR2TorchDataset, NLVR2Evaluator
from utils import load_obj_tsv
from operator import itemgetter

TINY_IMG_NUM = 10000
FAST_IMG_NUM = 5000

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

if args.tiny:
    topk = TINY_IMG_NUM
elif args.fast:
    topk = FAST_IMG_NUM
else:
    topk = -1

train_all_data = json.load(open("/home/achaud39/Abhishek/Experiments/robustness_using_counterfactuals/files/nlvr/processed/train_si_sp.json"))
valid_all_data = json.load(open("/home/achaud39/Abhishek/Experiments/robustness_using_counterfactuals/files/nlvr/processed/valid_si_sp.json"))
test_all_data = json.load(open("/home/achaud39/Abhishek/Experiments/robustness_using_counterfactuals/files/nlvr/processed/test_si_sp.json"))
print("Loaded all_data")
train_image_data = load_obj_tsv('/scratch/achaud39/nlvr/nlvr2/data/nlvr2_imgfeat/train_obj36.tsv', topk=topk)
print("Loaded train images")
valid_image_data = load_obj_tsv('/scratch/achaud39/nlvr/nlvr2/data/nlvr2_imgfeat/valid_obj36.tsv', topk=topk)
print("Loaded valid images")
test_image_data = load_obj_tsv('/scratch/achaud39/nlvr/nlvr2/data/nlvr2_imgfeat/test_obj36.tsv', topk=topk)
print("Loaded test images")



def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False, aug_item_ids = []) -> DataTuple:
    print("Splits ", splits)
    if splits == args.train:
        dset = NLVR2Dataset(splits, train_all_data, aug_item_ids)
        tset = NLVR2TorchDataset(dset, train_image_data)
    elif splits == args.valid:
        dset = NLVR2Dataset(splits, valid_all_data, aug_item_ids)
        tset = NLVR2TorchDataset(dset, valid_image_data)
    elif splits == args.test:
        dset = NLVR2Dataset(splits, test_all_data, aug_item_ids)
        tset = NLVR2TorchDataset(dset, test_image_data)

    evaluator = NLVR2Evaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator), tset


class NLVR2:
    def __init__(self):
        self.train_tuple, _ = get_tuple(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple, _ = get_tuple(
                args.valid, bs=valid_bsize,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None

        self.model = NLVR2Model()

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)

        # GPU options
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
        self.model = self.model.cuda()

        # Losses and optimizer
        self.mce_loss = nn.CrossEntropyLoss(ignore_index=-1)
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):

        
        aug_item_ids = []

        best_valid = 0.
        for epoch in range(args.epochs):
            print("Aug_items len", len(aug_item_ids))
            train_tuple, tset = get_tuple(
                args.train, bs=args.batch_size, shuffle=True, drop_last=True, aug_item_ids = aug_item_ids
            )

            dset, loader, evaluator = train_tuple
            # self.reset_optim(loader)
            iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, label, _, _) in iter_wrapper(enumerate(loader)):
                self.model.train()

                self.optim.zero_grad()
                feats, boxes, label = feats.cuda(), boxes.cuda(), label.cuda()
                logit = self.model(feats, boxes, sent)
                
                # COMPUTING LOSS
                loss = self.mce_loss(logit, label)

                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), 5.)
                self.optim.step()

                score, predict = logit.max(1)
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')


            if epoch >= args.n_pre and epoch < args.n_post:
                print("Starting evaluation")
                log_str += "Starting evaluation"

                # Getting the aug_item_ids using the original dataset
                eval_aug_item_ids , x = tset.get_aug_data()
                print(eval_aug_item_ids[0], x)
                eval_aug_item_ids = set(eval_aug_item_ids)
                aug_eval_dset = NLVR2Dataset(all_data = train_all_data, splits = args.train, aug_item_ids = eval_aug_item_ids, aug_eval = True) 
                aug_eval_tset = NLVR2TorchDataset(aug_eval_dset, train_image_data)
                aug_eval_evaluator = NLVR2Evaluator(aug_eval_dset)
                aug_eval_data_loader = DataLoader(
                    aug_eval_tset, batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers,
                    drop_last=False, pin_memory=True
                )

                # Perform evaluation of training set
                print("Created the eval dataset")
                aug_li, save_li = self.perform_eval(aug_eval_data_loader, x)
                print("Aug_li len", len(aug_li))
                # Add augmented data to training set
                aug_item_ids.extend(aug_li)

                # Only keep unique ids
                aug_item_ids = list(set(aug_item_ids))

            with open(self.output + "/log.log", 'a') as f:
                f.write(log_str)
                f.flush()

        self.save("LAST")

    def perform_eval(self, loader, x):
        self.model.eval()
        loss_li = []
        parent_li = []
        tag_li = []
        data_li = []
        final_aug_data_li = []
        save_li = []
        eval_loss = nn.CrossEntropyLoss(ignore_index=-1, reduce = False)

        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        
        for i, (ques_id, feats, boxes, sent, label, parents, tags) in iter_wrapper(enumerate(loader)):        
        # for i, datum_tuple in tqdm(enumerate(loader)):
            # ques_id, feats, boxes, sent, label, parents, tags = datum_tuple   # avoid handling target
            with torch.no_grad():
                # Change this to have tags and parents
                feats, boxes, label = feats.cuda(), boxes.cuda(), label.cuda()
                logit = self.model(feats, boxes, sent)
                
                # COMPUTING LOSS
                loss = eval_loss(logit, label).tolist()
                loss_li.extend(loss)

                parent_li.extend(parents)
                tag_li.extend(tags)
                data_li.extend(ques_id)
        
        assert len(loss_li) == len(parent_li)
        assert len(loss_li) == len(tag_li)
        assert len(loss_li) == len(data_li)

        req_len = x
        # get all the losses for each parent
        # Use this approach for performing argmax on parents
        if args.argmax_parents:
            valid_loss_for_each_parent_li = {}
            # print("Parents received", len(set(parent_li)))
            for idx in range(len(parent_li)):

                parent_id = parent_li[idx]
                data_item = data_li[idx]
                # print(idx, parent_id)
                loss_val = loss_li[idx]

                if parent_id not in valid_loss_for_each_parent_li:
                    valid_loss_for_each_parent_li[parent_id] = []
                valid_loss_for_each_parent_li[parent_id].append((loss_val, data_item))
            
            # For each parent get the max loss and corresponding data
            print("Number of parents", len(valid_loss_for_each_parent_li))
            for parent_id in valid_loss_for_each_parent_li:
                # li = sorted(valid_loss_for_each_parent_li[parent_id], key = lambda i: i[0], reverse = True)
                # aug_item = li[0][1]
                max_item =  max(valid_loss_for_each_parent_li[parent_id],key=itemgetter(0))
                aug_item = max_item[1]
                # aug_item['label'] = aug_item['label'].item()
                # aug_item['orig_label'] = aug_item['orig_label'].item()
                # print(aug_item['clip_id'])
                final_aug_data_li.append(aug_item)
                save_li.append(aug_item)

            print(len(final_aug_data_li), req_len, x)
            assert len(final_aug_data_li) == req_len
            return final_aug_data_li, save_li
        
        else: 

            # Here create dictionary of loses for each category, sort them in decreasing order
            valid_loss_for_each_tag_li = {}
            for idx in range(len(tag_li)):
                parent_id = parent_li[idx]
                data_item = data_li[idx]
                loss_val = loss_li[idx]
                tag = tag_li[idx]

                if tag not in valid_loss_for_each_tag_li:
                    valid_loss_for_each_tag_li[tag] = []
                valid_loss_for_each_tag_li[tag].append((loss_val, data_item))

            # Sort each dictionary in decreasing order
            for tag in valid_loss_for_each_tag_li:
                li = sorted(valid_loss_for_each_tag_li[tag], key = lambda i: i[0], reverse = True)
                valid_loss_for_each_tag_li[tag] = li

            
            print("Required length {}".format(req_len))
            prev = -1
            # Keep adding elements till length reaches the required length
            while len(final_aug_data_li) <= req_len:
                # logging.info(len(final_aug_data_li))

                for tag in valid_loss_for_each_tag_li:
                    # Is the list is not empty add first element
                    if len(valid_loss_for_each_tag_li[tag]) > 0:
                        aug_item = valid_loss_for_each_tag_li[tag][0][1]
                        # aug_item['label'] = aug_item['label'].item()
                        # aug_item['orig_label'] = aug_item['orig_label'].item()
                        final_aug_data_li.append(aug_item)
                        save_li.append(aug_item)
                        # Clear it from the list
                        del valid_loss_for_each_tag_li[tag][0]

                        if len(final_aug_data_li) == req_len:
                            break
                
                if len(final_aug_data_li) == req_len:
                    break

                # prev = len(final_aug_data_li)
            # print(len(final_aug_data_li), req_len)
            assert len(final_aug_data_li) == req_len
            # print(save_li[0])
            return final_aug_data_li, save_li



    def predict(self, eval_tuple: DataTuple, dump=None):
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, predict = logit.max(1)
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    quesid2ans[qid] = l
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        dset, loader, evaluator = eval_tuple
        quesid2ans = self.predict(eval_tuple, dump)
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)

    def reset_optim(self, loader):
        batch_per_epoch = len(loader)
        t_total = int(batch_per_epoch * args.epochs)
        print("Total Iters: %d" % t_total)
        
        self.optim = BertAdam(list(self.model.parameters()),
                                lr=args.lr,
                                warmup=0.1,
                                t_total=t_total)

if __name__ == "__main__":
    # Build Class
    nlvr2 = NLVR2()

    # Load Model
    if args.load is not None:
        nlvr2.load(args.load)
        
    print("Augmented Data Used ", args.adv_dataset)
    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'hidden' in args.test:
            nlvr2.predict(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'hidden_predict.csv')
            )
        elif 'test' in args.test or 'valid' in args.test:
            result = nlvr2.evaluate(
                get_tuple(args.test, bs=args.batch_size,
                          shuffle=False, drop_last=False),
                dump=os.path.join(args.output, '%s_predict.csv' % args.test)
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', nlvr2.train_tuple.dataset.splits)
        if nlvr2.valid_tuple is not None:
            print('Splits in Valid data:', nlvr2.valid_tuple.dataset.splits)
        else:
            print("DO NOT USE VALIDATION")
        nlvr2.train(nlvr2.train_tuple, nlvr2.valid_tuple)


