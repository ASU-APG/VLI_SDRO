import csv
import json
import os
import subprocess
from collections import defaultdict

test_file = "/home/achaud39/Abhishek/Experiments/lxmert/contrast_set_nlvr2.json"

with open(test_file, 'r') as f: 
    data = json.load(f)

for T in [1]:
    prediction_file_path = "/scratch/achaud39/lxmert_data/snap/nlvr2/nlvr2_lxr955_finetuned_contrast_results/test_predict.csv" # path to test_predict
    # acc_file_path = "test_accuracies/test_nlvr2_train_all_data_dataaug_{}_test_cat_accuracy.json".format(T) # path for category accuracy file

    # print("Reading from: {}, Writing to: {}".format(prediction_file_path, acc_file_path))

    identifier_result_d = dict()

    # if not os.path.exists(acc_file_path):
    #     continue 
    with open(prediction_file_path, mode='r') as infile:
        reader = csv.reader(infile)
        identifier_result_d = {rows[0]:rows[1] for rows in reader}
    correct_per_group = defaultdict(list)
    totals = {}
    corrects = {}
    
    # print(data)
    for item in data:
        identifier = item['identifier']
        group = item['identifier'][:-2]
        if identifier not in identifier_result_d: continue

        # if 'cf_tag' in item:
        #     tag = item['cf_tag']
        # elif 'sp_tag' in item:
        #     tag = item['sp_tag']
        # else: 
        #     tag = 'orig'
        tag = 'orig'
        # if tag == 'si_pronoun_substitution':
        #     print(item['identifier'])
        gt_label = int(item['label'])
        pred_label = identifier_result_d[identifier]

        if tag not in corrects:
            corrects[tag] = [0,0]
        if tag not in totals:
            totals[tag] = 0

        if pred_label == 'True':
            pred_label = 1
        else: 
            pred_label = 0

        if pred_label == gt_label:
            pred_label = 1
            correct_per_group[group].append(True)
        else: 
            pred_label = 0
            correct_per_group[group].append(False)

        corrects[tag][pred_label] += 1
        totals[tag] += 1

    cat_accuracy = {}
    avg_acc = 0
    for key in sorted(totals):
        cat_accuracy[key] = corrects[key][1]/totals[key]
        print(key, ":", cat_accuracy[key]*100)
        avg_acc += cat_accuracy[key]/len(totals)

    consistency = sum([all(g) for g in correct_per_group.values()]) / len(correct_per_group)
    print("Avg ", avg_acc)
    print(f"Consistency: {consistency}")
    # with open(acc_file_path, 'w') as outfile:
    #     json.dump(cat_accuracy, outfile, indent=4, sort_keys=True)
    #     json.dump({"avg_acc":avg_acc}, outfile, indent=4, sort_keys=True)
