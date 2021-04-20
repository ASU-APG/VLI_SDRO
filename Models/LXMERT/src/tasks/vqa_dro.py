# coding=utf-8
# Copyleft 2019 project LXRT.

import os
import collections
import operator
import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm

from param import args
from pretrain.qa_answer_table import load_lxmert_qa
from tasks.vqa_model import VQAModel
from tasks.vqa_data import VQADataset, VQATorchDataset, VQAEvaluator
from tasks.vqa_data_dro import VQADataset_DRO, VQATorchDataset_DRO
from utils import load_obj_tsv

DataTuple = collections.namedtuple("DataTuple", 'dataset loader evaluator')

# Load part of the dataset for fast checking.
# Notice that here is the number of images instead of the number of data,
# which means all related data to the images would be used.
TINY_IMG_NUM = 512
FAST_IMG_NUM = 5000

# The path to data and image features.
# The path to data and image features.
# VQA_DATA_ROOT = '/data/datasets/coco_text2img_gans/d2/'
# MSCOCO_IMGFEAT_ROOT = #'/data/datasets/d2_feats/'

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

if args.tiny:
    topk = TINY_IMG_NUM
elif args.fast:
    topk = FAST_IMG_NUM
else:
    topk = None

image_data = dict()
for split in ['train', 'minival', 'nominival']:
    load_topk = 5000 if (split == 'minival' and topk is None) else topk
    img_data = load_obj_tsv(
        os.path.join(MSCOCO_IMGFEAT_ROOT, '%s_obj36.tsv' % (SPLIT2NAME[split])),
        topk= load_topk,
        fp16=args.fp16)
    image_data[split] = img_data

    


def get_data_tuple_valid(splits: str, bs:int, shuffle=False, drop_last=False, aug_item_ids = []) -> DataTuple:
    print("Aug_items in get_data_tupel ", len(aug_item_ids))
    dset = VQADataset(splits)
    tset = VQATorchDataset(dset)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator), tset


def get_data_tuple_train(splits: str, bs:int, shuffle=False, drop_last=False, aug_item_ids = []) -> DataTuple:
    print("Aug_items in get_data_tuple ", len(aug_item_ids))
    dset = VQADataset_DRO(splits, aug_item_ids)
    tset = VQATorchDataset_DRO(dset, image_data)
    evaluator = VQAEvaluator(dset)
    data_loader = DataLoader(
        tset, batch_size=bs,
        shuffle=shuffle, num_workers=args.num_workers,
        drop_last=drop_last, pin_memory=True
    )

    return DataTuple(dataset=dset, loader=data_loader, evaluator=evaluator), tset

class VQA:
    def __init__(self):
        # Datasets
        self.train_tuple, _ = get_data_tuple_train(
            args.train, bs=args.batch_size, shuffle=True, drop_last=True
        )
        if args.valid != "":
            self.valid_tuple, _ = get_data_tuple_valid(
                args.valid, bs=1024,
                shuffle=False, drop_last=False
            )
        else:
            self.valid_tuple = None
        
        # Model
        self.model = VQAModel(self.train_tuple.dataset.num_answers)

        # Load pre-trained weights
        if args.load_lxmert is not None:
            self.model.lxrt_encoder.load(args.load_lxmert)
        if args.load_lxmert_qa is not None:
            load_lxmert_qa(args.load_lxmert_qa, self.model,
                           label2ans=self.train_tuple.dataset.label2ans)

        # Transfer model to GPU before apex.
        self.model = self.model.cuda()

        # Loss and Optimizer
        self.bce_loss = nn.BCEWithLogitsLoss()
        if 'bert' in args.optim:
            batch_per_epoch = len(self.train_tuple.loader)
            t_total = int(batch_per_epoch * args.epochs)
            print("BertAdam Total Iters: %d" % t_total)
            from lxrt.optimization import BertAdam
            self.optim = BertAdam(list(self.model.parameters()),
                                  lr=args.lr,
                                  warmup=0.1,
                                  t_total=t_total)
        else:
            self.optim = args.optimizer(self.model.parameters(), args.lr)


        # Half Precision 
        if args.fp16:
            try:
                from apex import amp
            except ImportError:
                raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
            self.model, self.optim = amp.initialize(self.model, self.optim, opt_level='O2')
        
        # GPU options
        if args.multiGPU:
            self.model.lxrt_encoder.multi_gpu()
        
        # Output Directory
        self.output = args.output
        os.makedirs(self.output, exist_ok=True)

    def train(self, train_tuple, eval_tuple):
        # dset, loader, evaluator = train_tuple
        # iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)
        
        aug_item_ids = []
        best_valid = 0.
        for epoch in range(args.epochs):

            print("Aug_items len", len(aug_item_ids))
            train_tuple, tset = get_data_tuple_train(
                args.train, bs=args.batch_size, shuffle=True, drop_last=True, aug_item_ids = aug_item_ids
            )
            dset, loader, evaluator = train_tuple
            self.reset_optim(loader)
            iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)

            quesid2ans = {}
            for i, (ques_id, feats, boxes, sent, target, _, _) in iter_wrapper(enumerate(loader)):

                self.model.train()
                self.optim.zero_grad()

                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent)
                assert logit.dim() == target.dim() == 2
                loss = self.bce_loss(logit, target)
                loss = loss * logit.size(1)

                if args.fp16:
                    try:
                        from apex import amp
                    except ImportError:
                        raise ImportError("Please install apex from https://www.github.com/nvidia/apex to use fp16 training.")
                    with amp.scale_loss(loss, self.optim) as scaled_loss:
                        scaled_loss.backward()
                    torch.nn.utils.clip_grad_norm_(amp.master_params(self.optim), 5.)
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 5.)

                self.optim.step()

                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans

            log_str = "\nEpoch %d: Train %0.2f\n" % (epoch, evaluator.evaluate(quesid2ans) * 100.)

            if self.valid_tuple is not None:  # Do Validation
                valid_score = self.evaluate(eval_tuple)
                if valid_score > best_valid:
                    best_valid = valid_score
                    self.save("BEST")

                log_str += "Epoch %d: Valid %0.2f\n" % (epoch, valid_score * 100.) + \
                           "Epoch %d: Best %0.2f\n" % (epoch, best_valid * 100.)

            print(log_str, end='')

            if epoch >= args.N_pre and epoch < args.N_post:
                print("Starting evaluation")
                log_str += "Starting evaluation"

                # Getting the aug_item_ids using the original dataset
                eval_aug_item_ids , x = tset.get_aug_data()
                # print(eval_aug_item_ids[:5], x)
                eval_aug_item_ids = set(eval_aug_item_ids)
                # print(len(eval_aug_item_ids))
                aug_eval_dset = VQADataset_DRO(splits = args.train, aug_item_ids = eval_aug_item_ids, aug_eval = True) 
                aug_eval_tset = VQATorchDataset_DRO(aug_eval_dset, image_data)
                # aug_eval_evaluator = VQAEvaluator(aug_eval_dset)
                aug_eval_data_loader = DataLoader(
                    aug_eval_tset, batch_size=args.batch_size,
                    shuffle=False, num_workers=args.num_workers,
                    drop_last=False, pin_memory=True
                )

                # Perform evaluation of training set
                print("Created the eval dataset")
                aug_li = []
                aug_li = self.perform_eval(aug_eval_data_loader, x)

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
        eval_loss = nn.BCEWithLogitsLoss(reduction='none')

        iter_wrapper = (lambda x: tqdm(x, total=len(loader))) if args.tqdm else (lambda x: x)


        for i, (ques_id, feats, boxes, sent, target, parents, tags) in iter_wrapper(enumerate(loader)):        
        # for i, datum_tuple in tqdm(enumerate(loader)):
            # ques_id, feats, boxes, sent, label, parents, tags = datum_tuple   # avoid handling target
            with torch.no_grad():
                # Change this to have tags and parents
                feats, boxes, target = feats.cuda(), boxes.cuda(), target.cuda()
                logit = self.model(feats, boxes, sent) 
                assert logit.dim() == target.dim() == 2
                # COMPUTING LOSS
                loss = eval_loss(logit, target)
                loss = torch.sum(loss, dim=1)
                # print(loss.size())
                # loss = loss * logit.size(1)
                # print(loss.size(),"--", logit.size()) # 32 * 3129 , 32 * 2, 32*1
                loss = loss.tolist()
                loss_li.extend(loss)
                # print(len(loss))
                
                assert len(loss) == len(parents)
                parent_li.extend(parents)
                tag_li.extend(tags)
                data_li.extend(ques_id)
        
        assert len(loss_li) == len(parent_li)
        assert len(loss_li) == len(tag_li)
        assert len(loss_li) == len(data_li)
        # print(loss_li[0], parent_li[0], tag_li[0], data_li[0])
        req_len = x
        if args.argmax_parents:
            valid_loss_for_each_parent_li = {}
            # print("Parents received", len(set(parent_li)))
            for idx in range(len(parent_li)):

                parent_id = parent_li[idx].item()
                data_item = data_li[idx].item()
                loss_val = loss_li[idx]
                tag = tag_li[idx]

                if parent_id not in valid_loss_for_each_parent_li:
                    valid_loss_for_each_parent_li[parent_id] = []
                valid_loss_for_each_parent_li[parent_id].append((loss_val, data_item))
            
            # For each parent get the max loss and corresponding data
            print("Number of parents", len(valid_loss_for_each_parent_li))
            for parent_id in valid_loss_for_each_parent_li:
                # li = sorted(valid_loss_for_each_parent_li[parent_id], key = lambda i: i[0], reverse = True)
                # aug_item = li[0][1]
                max_item =  max(valid_loss_for_each_parent_li[parent_id],key=operator.itemgetter(0))
                aug_item = max_item[1]
                # aug_item['label'] = aug_item['label'].item()
                # aug_item['orig_label'] = aug_item['orig_label'].item()
                # print(aug_item['clip_id'])
                final_aug_data_li.append(aug_item)
                save_li.append(aug_item)

            # print(len(final_aug_data_li), req_len, x)
            assert len(final_aug_data_li) >= req_len
            return final_aug_data_li
        else: 
            # Here create dictionary of loses for each category, sort them in decreasing order
            valid_loss_for_each_tag_li = {}
            for idx in range(len(tag_li)):
                parent_id = parent_li[idx].item()
                data_item = data_li[idx].item()
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

                        if len(final_aug_data_li) >= req_len:
                            break
                
                if len(final_aug_data_li) >= req_len:
                    break

                # prev = len(final_aug_data_li)
            # print(len(final_aug_data_li), req_len)
            assert len(final_aug_data_li) >= req_len            
            return final_aug_data_li



    def predict(self, eval_tuple: DataTuple, dump=None):
        """
        Predict the answers to questions in a data split.

        :param eval_tuple: The data tuple to be evaluated.
        :param dump: The path of saved file to dump results.
        :return: A dict of question_id to answer.
        """
        self.model.eval()
        dset, loader, evaluator = eval_tuple
        quesid2ans = {}
        for i, datum_tuple in enumerate(loader):
            ques_id, feats, boxes, sent = datum_tuple[:4]   # Avoid seeing ground truth
            with torch.no_grad():
                feats, boxes = feats.cuda(), boxes.cuda()
                logit = self.model(feats, boxes, sent)
                score, label = logit.max(1)
                for qid, l in zip(ques_id, label.cpu().numpy()):
                    ans = dset.label2ans[l]
                    quesid2ans[qid.item()] = ans
        if dump is not None:
            evaluator.dump_result(quesid2ans, dump)
        return quesid2ans

    def evaluate(self, eval_tuple: DataTuple, dump=None):
        """Evaluate all data in data_tuple."""
        quesid2ans = self.predict(eval_tuple, dump)
        return eval_tuple.evaluator.evaluate(quesid2ans)

    @staticmethod
    def oracle_score(data_tuple):
        dset, loader, evaluator = data_tuple
        quesid2ans = {}
        for i, (ques_id, feats, boxes, sent, target) in enumerate(loader):
            _, label = target.max(1)
            for qid, l in zip(ques_id, label.cpu().numpy()):
                ans = dset.label2ans[l]
                quesid2ans[qid.item()] = ans
        return evaluator.evaluate(quesid2ans)

    def save(self, name):
        torch.save(self.model.state_dict(),
                   os.path.join(self.output, "%s.pth" % name))

    def load(self, path):
        print("Load model from %s" % path)
        state_dict = torch.load("%s.pth" % path)
        self.model.load_state_dict(state_dict)


    def reset_optim(self, loader):
        from lxrt.optimization import BertAdam
        batch_per_epoch = len(loader)
        t_total = int(batch_per_epoch * args.epochs)
        print("Total Iters: %d" % t_total)
        
        self.optim = BertAdam(list(self.model.parameters()),
                                lr=args.lr,
                                warmup=0.1,
                                t_total=t_total)


if __name__ == "__main__":
    # Build Class
    vqa = VQA()

    # Load VQA model weights
    # Note: It is different from loading LXMERT pre-trained weights.
    if args.load is not None:
        vqa.load(args.load)

    # Test or Train
    if args.test is not None:
        args.fast = args.tiny = False       # Always loading all data in test
        if 'test' in args.test or 'minival' in args.test:
            vqa.predict(
                get_data_tuple_valid(args.test, bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'test_predict.json')
            )
        elif 'val' in args.test:    
            # Since part of valididation data are used in pre-training/fine-tuning,
            # only validate on the minival set.
            result = vqa.evaluate(
                get_data_tuple_valid('minival', bs=950,
                               shuffle=False, drop_last=False),
                dump=os.path.join(args.output, 'minival_predict.json')
            )
            print(result)
        else:
            assert False, "No such test option for %s" % args.test
    else:
        print('Splits in Train data:', vqa.train_tuple.dataset.splits)
        if vqa.valid_tuple is not None:
            print('Splits in Valid data:', vqa.valid_tuple.dataset.splits)
            print("Valid Oracle: %0.2f" % (vqa.oracle_score(vqa.valid_tuple) * 100))
        else:
            print("DO NOT USE VALIDATION")
        vqa.train(vqa.train_tuple, vqa.valid_tuple)


