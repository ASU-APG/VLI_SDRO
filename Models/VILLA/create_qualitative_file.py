import csv

test_file = "/data_1/data/uniter/ann/test_si_sp.json"

pred_file = "/data_1/data/uniter/finetune/villa_nlvr2/nlvr2_all_data_groupdro_20/results.csv"

with open(prediction_file_path, mode='r') as infile:
    reader = csv.reader(infile)
    identifier_result_d = {rows[0]:rows[1] for rows in reader}

for item in data:
        identifier = item['identifier']

        # if identifier not in identifier_result_d: continue

        tag = item['tag']

        gt_label = int(item['label'])
        pred_label = int(identifier_result_d[identifier] == 'True')
    

# "/data_1/data/uniter/finetune/villa_nlvr2/nlvr2_all_data_sampledro_20/results.csv"
# "/data_1/data/uniter/finetune/villa_nlvr2/nlvr2_dataaug_20/results.csv"
# "/data_1/data/uniter/finetune/villa_base_finetuned_recent_si_sp_results/results.csv"