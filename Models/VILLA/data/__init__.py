"""
Copyright (c) Microsoft Corporation.
Licensed under the MIT license.

"""
from .data import (TxtTokLmdb, DetectFeatLmdb,
                   ImageLmdbGroup, ConcatDatasetWithLens)
from .sampler import TokenBucketSampler
from .loader import PrefetchLoader, MetaLoader
from .vqa import VqaDataset, VqaEvalDataset, vqa_collate, vqa_eval_collate

from .vqa_stat import VqaDataset_STAT, vqa_collate_stat, VqaAugDataset_STAT, vqa_collate_aug_stat


from .ve import VeDataset, VeEvalDataset, ve_collate, ve_eval_collate
from .nlvr2 import (Nlvr2PairedDataset, Nlvr2PairedEvalDataset,
                    Nlvr2TripletDataset, Nlvr2TripletEvalDataset,
                    nlvr2_paired_collate, nlvr2_paired_eval_collate,
                    nlvr2_triplet_collate, nlvr2_triplet_eval_collate)
from .nlvr2_stat import (Nlvr2PairedDataset_STAT, nlvr2_paired_collate_stat, 
                    Nlvr2PairedDatasetEval_STAT, nlvr2_paired_collate_eval_stat)
from .itm import (TokenBucketSamplerForItm, ItmDataset,
                  itm_collate, itm_ot_collate,
                  ItmRankDataset, ItmValDataset, ItmEvalDataset,
                  ItmRankDatasetHardNegFromImage,
                  ItmRankDatasetHardNegFromText,
                  itm_rank_collate, itm_val_collate, itm_eval_collate,
                  itm_rank_hn_collate)
from .mlm import MlmDataset, mlm_collate
from .mrm import MrfrDataset, MrcDataset, mrfr_collate, mrc_collate
from .vcr import (VcrTxtTokLmdb, VcrDataset, VcrEvalDataset,
                  vcr_collate, vcr_eval_collate)
