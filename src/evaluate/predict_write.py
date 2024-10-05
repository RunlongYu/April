import pandas as pd
import torch
import os
from predict_group import  check_if_skip, test_lstm, test_pgrnn, test_pgrnn_extend

# lake_id = 'nhdhr_120018493'

for group_id in range(1,5):
    # group_id = 1
    # ids = pd.read_csv(f'../../data/utils/groups/vol_area/cluster_{group_id}.csv')
    ids = pd.read_csv(f'../../data/utils/groups/vol_area/evaluate/cluster_{group_id}.csv')
    ids = ids['nhdhr_id'].to_list()

    # ids = [lake_id]
    # save_path = f'../../data/results/results_lstm_train_one_obs_cluster_{group_id}.csv'
    # save_path = f'../../data/results/results_pgrnn_train_one_obs_cluster_{group_id}.csv'

    # save_path = f'../../data/results/results_pgrnn_train_one_obs_cluster_extend_{group_id}.csv'
    save_path = f'../../data/results/April_results_pgrnn_train_one_obs_cluster_extend_{group_id}.csv'
    if not os.path.exists('../../data/results'): 
        os.makedirs('../../data/results')
    results = []

    for count, lake_id in enumerate(ids):
        print("Lake ID:", lake_id)
        if_skip = check_if_skip(lake_id, group_id)
        if not if_skip:
            # results_lstm, results_pbmodel, obs_number = test_lstm(lake_id, group_id, False)

            # results_lstm_cluster, _, _ = test_lstm(lake_id, group_id, True)

            results_lstm, results_pbmodel, obs_number = test_pgrnn_extend(lake_id, group_id, False)
            results_lstm_cluster, _, _ = test_pgrnn_extend(lake_id, group_id, True)

            results_lstm = [val.item() if isinstance(val, torch.Tensor) else val for val in results_lstm]
            results_pbmodel = [val.item() if isinstance(val, torch.Tensor) else val for val in results_pbmodel]
            results_lstm_cluster = [val.item() if isinstance(val, torch.Tensor) else val for val in results_lstm_cluster]
            obs_number = [val.item() if isinstance(val, torch.Tensor) else val for val in obs_number]

            row = [lake_id] + results_lstm + results_lstm_cluster + results_pbmodel + obs_number
            results.append(row)
        else:
            print(f"Skip lake: {lake_id}")
            
        # except Exception as e:
        #     print(f"Skipping lake ID {lake_id} due to error: {e}")


    # columns = ['lake_id'] + ['lstm_total_rmse', 'lstm_mixed_rmse', 'lstm_upper_rmse', 'lstm_lower_rmse', 'total_PG_loss', 'stratified_PG_loss_upper', 'stratified_PG_loss_lower'] + ['ClusterLstm_total_rmse', 'ClusterLstm_mixed_rmse', 'ClusterLstm_upper_rmse', 'ClusterLstm_lower_rmse', 'ClusterLstm_total_PG_loss', 'ClusterLstm_stratified_PG_loss_upper', 'ClusterLstm_stratified_PG_loss_lower']+ ['PB_total_rmse', 'PB_mixed_rmse', 'PB_upper_rmse', 'PB_lower_rmse', 'PB_total_PG_loss', 'PB_stratified_PG_loss_upper', 'PB_stratified_PG_loss_lower'] 

    # columns = ['lake_id'] + ['pgrnn_total_rmse', 'pgrnn_mixed_rmse', 'pgrnn_upper_rmse', 'pgrnn_lower_rmse', 'pgrnn_total_PG_loss', 'pgrnn_stratified_PG_loss_upper', 'pgrnn_stratified_PG_loss_lower'] + ['ClusterPgrnn_total_rmse', 'ClusterPgrnn_mixed_rmse', 'ClusterPgrnn_upper_rmse', 'ClusterPgrnn_lower_rmse', 'ClusterPgrnn_total_PG_loss', 'ClusterPgrnn_stratified_PG_loss_upper', 'ClusterPgrnn_stratified_PG_loss_lower']+ ['PB_total_rmse', 'PB_mixed_rmse', 'PB_upper_rmse', 'PB_lower_rmse', 'PB_total_PG_loss', 'PB_stratified_PG_loss_upper', 'PB_stratified_PG_loss_lower'] 

    columns = ['lake_id'] + ['april_total_rmse', 'april_mixed_rmse', 'april_upper_rmse', 'april_lower_rmse', 'pgrnn_total_PG_loss', 'april_stratified_PG_loss_upper', 'april_stratified_PG_loss_lower'] + ['ClusterPgrnn_total_rmse', 'ClusterPgrnn_mixed_rmse', 'ClusterPgrnn_upper_rmse', 'ClusterPgrnn_lower_rmse', 'ClusterPgrnn_total_PG_loss', 'ClusterPgrnn_stratified_PG_loss_upper', 'ClusterPgrnn_stratified_PG_loss_lower']+ ['PB_total_rmse', 'PB_mixed_rmse', 'PB_upper_rmse', 'PB_lower_rmse', 'PB_total_PG_loss', 'PB_stratified_PG_loss_upper', 'PB_stratified_PG_loss_lower'] 

    columns = columns +['train_obs(days)','test_obs(days)']
    df = pd.DataFrame(results, columns=columns)

    df.to_csv(save_path, index=False)

    print(f"Results saved to {save_path}")
