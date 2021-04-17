import csv
import json
import os
import subprocess

# si_test_file = "/data/datasets/vqa/devval.json"
si_test_file = "/data_1/data/tgokhale/vl_textattacks/output/vqayesno_devval_textattack.json"

with open(si_test_file, 'r') as f: 
    data = json.load(f)


for T in [1]:
    # prediction_file_path = "/data_1/data/uniter/finetune/uniter_nlvr2/nlvr2_all_data_stmt_sampledro_{}_textattack_results/results.csv".format(T) # path to test_predict
    # acc_file_path = "test_accuracies/textattack/test_nlvr2_all_data_stmt_sampledro_{}_test_cat_accuracy.json".format(T) # path for category accuracy file
    prediction_file_path = "/data_1/data/uniter/finetune/uniter_vqa/train_vqa_baseline_orig/results_test/results_6000_all.json"
    acc_file_path = "test_accuracies/vqa/train_vqa_groupdro_20_textattack.json"

    print("Reading from: {}, Writing to: {}".format(prediction_file_path, acc_file_path))

    identifier_result_d = dict()

    # if not os.path.exists(acc_file_path):
    #     continue 
    with open(prediction_file_path, mode='r') as infile:
        prediction_results = json.load(infile)
        for item in prediction_results:
            ques_id = item['question_id']
            # print(ques_id)
            res = item['answer']

            identifier_result_d[ques_id] = res

    totals = {}
    corrects = {}
    # print(data)
    for item in data:
        identifier = item['question_id']

        # if identifier not in identifier_result_d: continue

        tag = item['attack']

        gt_label = item['label']
        if "yes" in gt_label:
            gt_label = "yes"
        else:
            gt_label = "no"
        pred_label = identifier_result_d[identifier]

        if tag not in corrects:
            corrects[tag] = 0
        if tag not in totals:
            totals[tag] = 0

        if pred_label == gt_label:
            corrects[tag] += 1
        totals[tag] += 1

    cat_accuracy = {}
    avg_acc = 0
    for key in sorted(totals):
        cat_accuracy[key] = corrects[key]/totals[key]
        print(key, ":", cat_accuracy[key]*100)
        avg_acc += cat_accuracy[key]/len(totals)

    print("Avg ", avg_acc)
    with open(acc_file_path, 'w') as outfile:
        json.dump(cat_accuracy, outfile, indent=4, sort_keys=True)
        json.dump({"avg_acc":avg_acc}, outfile, indent=4, sort_keys=True)
