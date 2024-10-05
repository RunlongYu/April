import pandas as pd
import torch
import torch.nn as nn
import numpy as np
from datetime import datetime
import sys
import csv
import os
sys.path.append('../../utils')
sys.path.append('../utils')
from evaluate_utils import predict_and_write_new
from train_utils import PretrainAprilCfg, FineTuneAprilCfg, set_seed, random_seed
from predict_group import check_if_skip, draw_one_lake



time = datetime.now().strftime("%Y%m%d_%H%M%S")
# save Train config
configs_pre = {}
configs_fine = {}
for cluster_id in range(1, 5):
    configs_pre[f'Pretrain_cluster_{cluster_id}'] = PretrainAprilCfg(cluster_id)
    configs_fine[f'FineTune_cluster_{cluster_id}'] = FineTuneAprilCfg(cluster_id)

cfg_save_path = f'../../results/{time}/cfg.txt'
os.makedirs(os.path.dirname(cfg_save_path), exist_ok=True)
with open(cfg_save_path, 'w') as f:
    f.write(f"Pre-train config April:\n")
    for config_name, config in configs_pre.items():
        f.write(f"Cluster ID: {config.cluster_id}\n")
        f.write(f"Train Epochs: {config.train_epochs}\n")
        f.write(f"Batch Size: {config.batch_size}\n")
        f.write(f"Learning Rate: {config.learning_rate}\n")
        f.write(f"Lambda Total: {config.lambda_total}\n")
        f.write(f"Lambda Stratified Epi: {config.lambda_stratified_epi}\n")
        f.write(f"Lambda Stratified Hypo: {config.lambda_stratified_hypo}\n")
        f.write(f"---------------------------------\n")
    f.write(f"\n ========================================================\n")
    f.write(f"\nFine-tune config April:\n")
    for config_name, config in configs_fine.items():
        f.write(f"Cluster ID: {config.cluster_id}\n")
        f.write(f"Train Epochs: {config.train_epochs}\n")
        f.write(f"Batch Size: {config.batch_size}\n")
        f.write(f"Learning Rate: {config.learning_rate}\n")
        f.write(f"Lambda Total: {config.lambda_total}\n")
        f.write(f"Lambda Stratified Epi: {config.lambda_stratified_epi}\n")
        f.write(f"Lambda Stratified Hypo: {config.lambda_stratified_hypo}\n")
        f.write(f"---------------------------------\n")
Model_list = ['pb', 'lstm', 'ealstm', 'transformer', 'tft', 'pgrnn', 'april']
columns = ['mixed_rmse', 'upper_rmse', 'lower_rmse', 'Mixed_PG_loss', 'Strat_PG_loss_upper', 'Strat_PG_loss_lower']
columns_1 = [f"c1_{col}" for col in columns]
columns_2 = [f"c2_{col}" for col in columns]
columns_3 = [f"c3_{col}" for col in columns]
columns_4 = [f"c4_{col}" for col in columns]

# total_columns = columns_1 + columns_2 + columns_3 + columns_4
total_columns = columns_1
random_seed = [40, 42, 44]
doc_sh = 0.0
results = []
for model_index in range(1,4):
    seed = random_seed[model_index - 1]
    set_seed(seed)
    avg_results_all_cluster = {'lstm': {}, 'pgrnn': {}, 'april': {}, 'pb': {}, 'ealstm': {}, 'transformer': {}, 'tft': {}}
    for group_id in range(1,2):
        avg_results = predict_and_write_new(time, group_id, model_index, doc_sh)
        print("avg_results:", avg_results)
        for model, values in avg_results.items():
            avg_results_all_cluster[model][group_id] = avg_results[model]

        # draw for lakes
        ids = pd.read_csv(f'../../data/utils/groups/vol_area/removed/cluster_{group_id}.csv')
        ids = ids['nhdhr_id'].to_list()
        for count, lake_id in enumerate(ids):
            print("Lake ID:", lake_id)

            lstm_model_path = f'../../models/41/seed={seed}/lstm/group_{group_id}/fine_tune/individual_train_on_obs/{lake_id}_lstm_fine_tune_train_on_obs'
            pril_model_path = f'../../models/41/seed={seed}/pgrnn/group_{group_id}/fine_tune/individual_train_on_obs/{lake_id}_pgrnn_fine_tune_train_on_obs'
            april_model_path = f'../../models/41/seed={seed}/pgrnn/group_{group_id}/fine_tune/individual_train_on_obs_extend/{lake_id}_pgrnn_fine_tune_train_on_obs_12k'
            model_dir_list = [lstm_model_path, pril_model_path, april_model_path]
            # save_path = ['LSTM', 'Pril', 'April']
            model_dir_list = [april_model_path]
            save_path = ['April']

            base_save = f'../../results/{time}/pics/seed={seed}/cluster_{group_id}/'
            if not os.path.exists(base_save):
                os.makedirs(base_save)
            for i, model_dir in enumerate(model_dir_list):
                if group_id == 2 and model_index == 1:
                    draw_one_lake(seed, model_dir, base_save + save_path[i], lake_id, count, True)
    
    avg_save_path = f'../../results/{time}/data/seed={seed}/avg_results_sedd={seed}.csv'
    os.makedirs(os.path.dirname(avg_save_path), exist_ok=True)
    rows = []
    for model in Model_list:
        row = []
        for group_id in range(1,2):
            row.extend(avg_results_all_cluster[model][group_id].tolist())
        rows.append(row)

    results.append(rows)
    print("total_columns:",len(total_columns))
    if len(rows[0]) == len(total_columns):
        df = pd.DataFrame(rows, columns=total_columns)
        df.insert(0, 'Model type', Model_list)
        df.to_csv(avg_save_path, index=False)
        print(f"Results saved to {avg_save_path}")
    else:
        print("Mismatch in row and column lengths, cannot create DataFrame")


# save mean and std
results_np = np.array(results)
print("results_np shape:", results_np.shape)

mean_results = np.mean(results_np, axis=0)
std_results = np.std(results_np, axis=0)

print("mean_results shape:", mean_results.shape)
mean_df = pd.DataFrame(mean_results, columns=total_columns)  # 从第二列开始，因为将添加模型名称
std_df = pd.DataFrame(std_results, columns=total_columns)

mean_df.insert(0, 'Model type', Model_list)
std_df.insert(0, 'Model type', Model_list)

mean_df.to_csv(f'../../results/{time}/data/mean_results.csv', index=False)
std_df.to_csv(f'../../results/{time}/data/std_results.csv', index=False)

print("Mean and standard deviation results have been saved.")
