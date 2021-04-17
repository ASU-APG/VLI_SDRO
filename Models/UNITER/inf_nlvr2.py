"""run inference of NLVR2 (single GPU only)"""
import argparse
import json
import os
from os.path import exists
from time import time
import numpy as np 
from torch.nn import functional as F
import torch
from torch.utils.data import DataLoader
from apex import amp
from horovod import torch as hvd
from tqdm import tqdm 
from data import (DetectFeatLmdb, TxtTokLmdb,
                  PrefetchLoader, TokenBucketSampler,
                  Nlvr2PairedEvalDataset, Nlvr2TripletEvalDataset,
                  nlvr2_paired_eval_collate, nlvr2_triplet_eval_collate)
from model.model import UniterConfig
from model.nlvr2 import (UniterForNlvr2Paired, UniterForNlvr2Triplet,
                         UniterForNlvr2PairedAttn)

from utils.misc import Struct
from utils.const import IMG_DIM, BUCKET_SIZE


def main(opts):
    hvd.init()
    device = torch.device("cuda")  # support single GPU only
    train_opts = Struct(json.load(open(f'{opts.train_dir}/log/hps.json')))

    if 'paired' in train_opts.model:
        EvalDatasetCls = Nlvr2PairedEvalDataset
        eval_collate_fn = nlvr2_paired_eval_collate
        if train_opts.model == 'paired':
            ModelCls = UniterForNlvr2Paired
        elif train_opts.model == 'paired-attn':
            ModelCls = UniterForNlvr2PairedAttn
        else:
            raise ValueError('unrecognized model type')
    elif train_opts.model == 'triplet':
        EvalDatasetCls = Nlvr2TripletEvalDataset
        ModelCls = UniterForNlvr2Triplet
        eval_collate_fn = nlvr2_triplet_eval_collate
    else:
        raise ValueError('unrecognized model type')

    img_db = DetectFeatLmdb(opts.img_db,
                            train_opts.conf_th, train_opts.max_bb,
                            train_opts.min_bb, train_opts.num_bb,
                            opts.compressed_db)
    txt_db = TxtTokLmdb(opts.txt_db, -1)
    dset = EvalDatasetCls(txt_db, img_db, train_opts.use_img_type)
    batch_size = (train_opts.val_batch_size if opts.batch_size is None
                  else opts.batch_size)
    sampler = TokenBucketSampler(dset.lens, bucket_size=BUCKET_SIZE,
                                 batch_size=batch_size, droplast=False)
    eval_dataloader = DataLoader(dset, batch_sampler=sampler,
                                 num_workers=opts.n_workers,
                                 pin_memory=opts.pin_mem,
                                 collate_fn=eval_collate_fn)
    eval_dataloader = PrefetchLoader(eval_dataloader)

    # Prepare model
    ckpt_file = f'{opts.train_dir}/ckpt/model_step_{opts.ckpt}.pt'
    checkpoint = torch.load(ckpt_file)
    model_config = UniterConfig.from_json_file(
        f'{opts.train_dir}/log/model.json')
    model = ModelCls(model_config, img_dim=IMG_DIM)
    model.init_type_embedding()
    model.load_state_dict(checkpoint, strict=False)
    model.to(device)
    model = amp.initialize(model, enabled=opts.fp16, opt_level='O2')

    results = evaluate(model, eval_dataloader, device)
    # write results
    if not exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    with open(f'{opts.output_dir}/results.csv', 'w') as f:
        for id_, ans,target, loss  in results:
            f.write(f'{id_},{ans},{target},{loss}\n')
    print(f'all results written')


@torch.no_grad()
def evaluate(model, eval_loader, device):
    print("start running evaluation...")
    model.eval()
    n_ex = 0
    st = time()
    results = []
    denom_count = 0
    acc = 0
    for i, batch in tqdm(enumerate(eval_loader), ascii=True):
        # print(i)
        qids = batch['qids']
        # added by tgokhale 
        targets = batch['targets'].tolist()
        # ----
        # del batch['targets']
        del batch['qids']
        scores = model(batch, compute_loss=False)
        loss = F.cross_entropy(scores, batch['targets'], reduce=False)
        # val_loss += loss.sum().item()
        answers = [1 if i == 1 else 0
                   for i in scores.max(dim=-1, keepdim=False
                                       )[1].cpu().tolist()]
        predictions = [1 if i == 1 else 0 
                       for i in scores.max(dim=-1, keepdim=False
                                       )[1].cpu().tolist()]
        denom_count += len(targets)
        # print(targets)
        # print(predictions)
        acc += np.equal(targets, predictions).sum()
        
        results.extend(zip(qids, answers, targets, loss))
        n_results = len(results)
        # print(f'{n_results}/{len(eval_loader.dataset)} answers predicted')
        n_ex += len(qids)

    tot_time = time()-st
    print("Acc ", acc)
    print("test accuracy:", acc/denom_count)
    model.train()
    print(f"evaluation finished in {int(tot_time)} seconds "
          f"at {int(n_ex/tot_time)} examples per second")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Required parameters
    parser.add_argument("--txt_db",
                        type=str, required=True,
                        help="The input train corpus.")
    parser.add_argument("--img_db",
                        type=str, required=True,
                        help="The input train images.")

    parser.add_argument('--compressed_db', action='store_true',
                        help='use compressed LMDB')
    parser.add_argument("--batch_size", type=int,
                        help="batch size for evaluation")
    parser.add_argument('--n_workers', type=int, default=4,
                        help="number of data workers")
    parser.add_argument('--pin_mem', action='store_true',
                        help="pin memory")
    parser.add_argument('--fp16', action='store_true',
                        help="fp16 inference")

    parser.add_argument("--train_dir", type=str, required=True,
                        help="The directory storing NLVR2 finetuning output")
    parser.add_argument("--ckpt", type=int, required=True,
                        help="specify the checkpoint to run inference")
    parser.add_argument("--output_dir", type=str, required=True,
                        help="The output directory where the prediction "
                             "results will be written.")
    args = parser.parse_args()

    main(args)
