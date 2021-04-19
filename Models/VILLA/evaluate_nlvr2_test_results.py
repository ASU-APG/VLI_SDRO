import csv
import json
import os
import subprocess

test_file = "/data_1/data/uniter/ann/test_si_sp.json"

with open(test_file, 'r') as f: 
    data = json.load(f)

for T in [20]:
    # prediction_file_path = "/data_1/data/uniter/finetune/uniter_nlvr2/nlvr2_all_data_stmt_sampledro_{}_textattack_results/results.csv".format(T) # path to test_predict
    # acc_file_path = "test_accuracies/textattack/test_nlvr2_all_data_stmt_sampledro_{}_test_cat_accuracy.json".format(T) # path for category accuracy file
    prediction_file_path = "/data_1/data/uniter/finetune/villa_nlvr2/nlvr2_all_data_groupdro_20/results.csv"
    acc_file_path = "test_accuracies/sp_si/villa_base_finetuned_nlvr2_train_only_sp_20.json"

    print("Reading from: {}, Writing to: {}".format(prediction_file_path, acc_file_path))

    identifier_result_d = dict()

    # if not os.path.exists(acc_file_path):
    #     continue 
    with open(prediction_file_path, mode='r') as infile:
        reader = csv.reader(infile)
        identifier_result_d = {rows[0]:rows[1] for rows in reader}

    totals = {}
    corrects = {}
    # print(data)
    for item in data:
        identifier = item['identifier']

        # if identifier not in identifier_result_d: continue

        tag = item['tag']

        gt_label = int(item['label'])
        pred_label = int(identifier_result_d[identifier])

        if tag not in corrects:
            corrects[tag] = [0,0]
        if tag not in totals:
            totals[tag] = 0

        # if pred_label == 'True':
        #     pred_label = 1
        # else: 
        #     pred_label = 0

        if pred_label == gt_label:
            pred_label = 1
        else: 
            pred_label = 0

        corrects[tag][pred_label] += 1
        totals[tag] += 1

    cat_accuracy = {}
    avg_acc = 0
    for key in sorted(totals):
        cat_accuracy[key] = corrects[key][1]/totals[key]
        print(key, ":", cat_accuracy[key]*100)
        avg_acc += cat_accuracy[key]/len(totals)

    print("Avg ", avg_acc)
    # with open(acc_file_path, 'w') as outfile:
    #     json.dump(cat_accuracy, outfile, indent=4, sort_keys=True)
    #     json.dump({"avg_acc":avg_acc}, outfile, indent=4, sort_keys=True)
