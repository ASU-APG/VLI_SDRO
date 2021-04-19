import csv
import json
import os
import subprocess

test_file = "/data_1/data/uniter/ann/test_si_sp.json"

with open(test_file, 'r') as f: 
    data = json.load(f)

prediction_file_path = "/data_1/data/uniter/finetune/uniter_nlvr2/nlvr2_all_data_stmt_groupdro_20/results.csv"

identifier_result_d = dict()

with open(prediction_file_path, mode='r') as infile:
    reader = csv.reader(infile)
    identifier_result_d = {rows[0]:rows[1] for rows in reader}

totals = {}
corrects = {}

for item in data:
    identifier = item['identifier']
    tag = item['tag']
    gt_label = int(item['label'])
    pred_label = int(identifier_result_d[identifier])

    if tag not in corrects:
        corrects[tag] = [0,0]
    if tag not in totals:
        totals[tag] = 0

    if pred_label == gt_label:
        pred_label = 1
    else: 
        pred_label = 0

    corrects[tag][pred_label] += 1
    totals[tag] += 1

cat_accuracy = {}
avg_acc, avg_si, avg_sp = 0, 0, 0
for key in sorted(totals):
    cat_accuracy[key] = corrects[key][1]/totals[key]
    # print(key, ":", cat_accuracy[key]*100)
    avg_acc += cat_accuracy[key]/len(totals)
    if "si_" in key:
        avg_si += (cat_accuracy[key])/7
    if "sp_" in key:
        avg_sp += cat_accuracy[key]/6

print("Avg SI Accuracy {}".format(avg_si))
print("Avg SP Accuracy {}".format(avg_sp))
print("Original Accuracy {}".format(cat_accuracy['orig']))