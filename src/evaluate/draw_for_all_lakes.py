import pandas as pd
import torch
import os
from predict_group import check_if_skip, draw_one_lake

random_seed = [40, 42, 44]
# 读取 id 列表 
for model_index in range(1,4):
    for group_id in range(1,5):
        # group_id = 2
        ids = pd.read_csv(f'../../data/utils/groups/vol_area/evaluate/cluster_{group_id}.csv')
        ids = ids['nhdhr_id'].to_list()
        seed = random_seed[model_index - 1]
        for count, lake_id in enumerate(ids):
            print("Lake ID:", lake_id)

            lstm_model_path = f'../../models/41/seed={seed}/lstm/group_{group_id}/fine_tune/individual_train_on_obs/{lake_id}_lstm_fine_tune_train_on_obs'
            pril_model_path = f'../../models/41/seed={seed}/pgrnn/group_{group_id}/fine_tune/individual_train_on_obs/{lake_id}_pgrnn_fine_tune_train_on_obs'
            april_model_path = f'../../models/41/seed={seed}/pgrnn/group_{group_id}/fine_tune/individual_train_on_obs_extend/{lake_id}_pgrnn_fine_tune_train_on_obs_12k'


            model_dir_list = [lstm_model_path, pril_model_path, april_model_path]
            save_path = ['LSTM', 'Pril', 'April']

            base_save = f'../../results/pics/seed={seed}/cluster_{group_id}/'
            for i, model_dir in enumerate(model_dir_list):
                draw_one_lake(seed, model_dir, base_save + save_path[i], lake_id, count, True)