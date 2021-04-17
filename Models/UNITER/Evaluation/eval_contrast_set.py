import argparse
import json
from collections import defaultdict


if __name__ == '__main__':


    # prediction file is expected to be a text file with separate line per prediction,
    # with identifier and prediction (as 0/1) separated by a comma.
    # e.g.
    # test1-769-1-0-1,1
    # test1-769-1-0-2,0
    # test1-256-2-0-1,0
    # test1-256-2-0-2,1

    lines = list(open('/data_1/data/uniter/finetune/uniter_nlvr2/nlvr2_all_data_stmt_sampledro_20_contrast_set_results/results.csv', 'r'))
    predictions = [line.split(',')[:2] for line in lines]
    pred_per_identifier = {identifier: pred.strip() for identifier, pred in predictions}

    n_total = 0
    n_correct = 0
    correct_per_group = defaultdict(list)

    for line in open('/data_1/data/uniter/ann/contrast_set_nlvr2.jsonl', 'rt'):
        ex = json.loads(line)

        identifier = ex['identifier']
        group = ex['identifier'][:-2]
        gold_label = int(ex['label'])
        correct = False

        

        if identifier in pred_per_identifier:
            n_total += 1
            # print(pred_per_identifier[identifier], )
            pred = int(pred_per_identifier[identifier])
            # print(pred, gold_label)
            correct = pred == gold_label

            if correct:
                n_correct += 1
            correct_per_group[group].append(correct)
        else:
            # prediction not found
            pass

        
    print(f"n_correct : {n_correct}")
    print(f"n_total : {n_total}")
    acc = n_correct / n_total
    consistency = sum([all(g) for g in correct_per_group.values()]) / len(correct_per_group)
    print(f"Accuracy: {acc}")
    print(f"Consistency: {consistency}")