# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import json
import collections

from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader

from param import args
from tasks.nlvr2_model import NLVR2Model
from tasks.nlvr2_data_dro import NLVR2Dataset_DRO, NLVR2TorchDataset_DRO, NLVR2Dataset_DRO_EVAL, NLVR2Evaluator_DRO

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')


# def get_tuple(splits: str, bs:int, shuffle=False, drop_last=False) -> DataTuple:
#     print("Splits ", splits)
#     dset = NLVR2Dataset_DRO(splits)
#     tset = NLVR2TorchDataset_DRO(dset)
#     evaluator = NLVR2Dataset_DRO_EVAL(dset)
#     data_loader = DataLoader(
#         tset, batch_size=bs,
#         shuffle=shuffle, num_workers=args.num_workers,
#         drop_last=drop_last, pin_memory=True
#     )

#     return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator)


class NLVR2:
    def __init__(self, loader):
        self.train_tuple = None
        if args.valid != "":
            valid_bsize = 2048 if args.multiGPU else 512
            self.valid_tuple = None
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
            batch_per_epoch = len(loader)
            # batch_per_epoch = args.batch_size
            t_total = int(batch_per_epoch * args.epochs)
            print("Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(list(self.model.parameters()), args.lr)

        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, dset, tset, evaluator, loader, eval_tuple):
      
        best_valid = 0.
        for epoch in range(args.epochs):
            aug_li_save_file = "{}/augmented_items_added_{}.json".format(self.output, epoch)
            loader = DataLoader(
                tset, batch_size=args.batch_size,
                shuffle=True, num_workers=args.num_workers,
                drop_last=True, pin_memory=True
            )
            iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, label, parents, tags) in iter_wrapper(enumerate(loader)):
                # print(ques_id[0], tags[0])
                self.model.train()

                self.optim.zero_grad()
                # Change this to have tags and parents
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

            # print(quesid2ans)
            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            # if self.valid_tuple is not None:  # Do Validation
            valid_score = self.evaluate()
            if valid_score > best_valid:
                best_valid = valid_score
                self.save("BEST")

            log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str)


            # perform eval if epoch >= args.n_pre and epoch < args.n_post:
            if epoch >= args.n_pre and epoch < args.n_post:
                log_str += "Starting evaluation"
                eval_set = tset.get_eval_data() 
                eval_loader = DataLoader(
                    eval_set, batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers,
                    drop_last=False, pin_memory=True
                )
                # Perform evaluation of training set
                aug_li, save_li = self.perform_eval(eval_loader, eval_set.x)

                with open(aug_li_save_file, 'w') as outfile:
                    json.dump(save_li, outfile, indent=4)

                # Add augmented data to training set
                tset.add_aug_data(aug_li)


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
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent, label, parents, tags, _datums = datum_tuple   # avoid handling target
            with torch.no_grad():
                # Change this to have tags and parents
                feats, boxes, label = feats.cuda(), boxes.cuda(), label.cuda()
                logit = self.model(feats, boxes, sent)
                
                # COMPUTING LOSS
                loss = eval_loss(logit, label).tolist()
                # print(len(loss), len(parents), len(datums))
                loss_li.extend(loss)
                
                datums = [{key:value[index] for key,value in _datums.items()} for index in range(max(map(len,_datums.values())))]

                # print(len(loss), len(parents), len(datums))
                parent_li.extend(parents)
                tag_li.extend(tags)
                data_li.extend(datums)
        
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
            # print("Number of parents", len(valid_loss_for_each_parent_li))
            for parent_id in valid_loss_for_each_parent_li:
                li = sorted(valid_loss_for_each_parent_li[parent_id], key = lambda i: i[0], reverse = True)
                aug_item = li[0][1]
                aug_item['label'] = aug_item['label'].item()
                aug_item['orig_label'] = aug_item['orig_label'].item()
                # print(aug_item['clip_id'])
                final_aug_data_li.append(aug_item)
                save_li.append(aug_item)

            # print(len(final_aug_data_li), req_len, x)
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
                        aug_item['label'] = aug_item['label'].item()
                        aug_item['orig_label'] = aug_item['orig_label'].item()
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



    def predict(self, dump=None):
        self.model.eval()
        # dset, loader, evaluator = eval_tuple
        dset = NLVR2Dataset_DRO(args.valid)
        tset = NLVR2TorchDataset_DRO(dset)
        evaluator = NLVR2Evaluator_DRO(dset)
        loader = DataLoader(
            tset, batch_size=args.batch_size,
            shuffle=False, num_workers=args.num_workers,
            drop_last=False, pin_memory=True
        )
        quesid2ans = {}
        # print("Loader ", len(loader))
        for i, datum_tuple in enumerate(loader):
            # print(i)
            ques_id, feats, boxes, sent = datum_tuple[:4]   # avoid handling target
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, predict = logit.max(1)
                # print("Score ", score.shape)
                for qid, l in zip(ques_id, predict.cpu().numpy()):
                    # print("Qid ", qid, l)
                    quesid2ans[qid] = l
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return evaluator.evaluate(quesid2ans)

    def evaluate(self, dump=None):
        # dset, loader, evaluator = eval_tuple
        return self.predict()

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


if __name__ == "__main__":
    # Build Class
    dset = NLVR2Dataset_DRO(args.train)
    tset = NLVR2TorchDataset_DRO(dset)
    evaluator = NLVR2Evaluator_DRO(dset)
    loader = DataLoader(
        tset, batch_size=args.batch_size,
        shuffle=True, num_workers=args.num_workers,
        drop_last=True, pin_memory=True
    )
    nlvr2 = NLVR2(loader)

    # Load Model
    if args.load is not None:
        nlvr2.load(args.load)
        
    print("Augmented Data Used ", args.adv_dataset)
    print("CMD Params")
    print("N {}, n_pre {}, n_post {}, T {} ".format(args.epochs, args.n_pre, args.n_post, args.T))
    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        # if 'hidden' in args.test:
        #     nlvr2.predict(
        #         get_tuple(args.test, bs=args.batch_size,
        #                   shuffle=False, drop_last=False),
        #         dump=os.path.join(args.output, 'hidden_predict.csv')
        #     )
        # elif 'test' in args.test or 'valid' in args.test:
        #     result = nlvr2.evaluate(
        #         get_tuple(args.test, bs=args.batch_size,
        #                   shuffle=False, drop_last=False),
        #         dump=os.path.join(args.output, '%s_predict.csv' % args.test)
        #     )
        #     print(result)
        # else:
        #     assert False, "No such test option for %s" % args.test
    else:
        # print('Splits in Train data:', nlvr2.train_tuple.dataset.splits)
        # if nlvr2.valid_tuple is not None:
        #     print('Splits in Valid data:', nlvr2.valid_tuple.dataset.splits)
        # else:
        #     print("DO NOT USE VALIDATION")
        nlvr2.train(dset, tset, evaluator, loader, nlvr2.valid_tuple)


