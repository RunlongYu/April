import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import sys
import csv
import os
sys.path.append('../../utils')
sys.path.append('../utils')
sys.path.append('../../evaluate')
sys.path.append('../evaluate')
sys.path.append('../../data')
sys.path.append('../data')
sys.path.append('../../models')
sys.path.append('../models')
from lakePreprocess import DROP_COLUMNS, USE_FEATURES_COLUMNS, FLUX_COLUMNS, USE_FEATURES_COLUMNS_LAYER
from pytorch_data_operations import buildManyLakeDataByIds, calculate_total_DOC_conservation_loss, calculate_stratified_DOC_conservation_loss, buildManyLakeDataByIds_ForTFT
from train_utils import MyEALSTM, TransformerModel, TransformerConfig, MyLSTM, set_seed, random_seed
from predict_group import test_lstm, test_pgrnn, test_pgrnn_extend, get_obs_number


columns_lstm = ['lstm_total_rmse', 'lstm_mixed_rmse', 'lstm_upper_rmse', 'lstm_lower_rmse', 'total_PG_loss', 'stratified_PG_loss_upper', 'stratified_PG_loss_lower'] 
columns_pgrnn = ['pgrnn_total_rmse', 'pgrnn_mixed_rmse', 'pgrnn_upper_rmse', 'pgrnn_lower_rmse', 'pgrnn_total_PG_loss', 'pgrnn_stratified_PG_loss_upper', 'pgrnn_stratified_PG_loss_lower']
columns_april = ['april_total_rmse', 'april_mixed_rmse', 'april_upper_rmse', 'april_lower_rmse', 'april_total_PG_loss', 'april_stratified_PG_loss_upper', 'april_stratified_PG_loss_lower']
columns_pb = ['PB_total_rmse', 'PB_mixed_rmse', 'PB_upper_rmse', 'PB_lower_rmse', 'PB_total_PG_loss', 'PB_stratified_PG_loss_upper', 'PB_stratified_PG_loss_lower']
columns_ealstm = ['ealstm_total_rmse', 'ealstm_mixed_rmse', 'ealstm_upper_rmse', 'ealstm_lower_rmse', 'ealstm_total_PG_loss', 'ealstm_stratified_PG_loss_upper', 'ealstm_stratified_PG_loss_lower']
columns_transformer = ['TF_total_rmse', 'TF_mixed_rmse', 'TF_upper_rmse', 'TF_lower_rmse', 'TF_total_PG_loss', 'TF_stratified_PG_loss_upper', 'TF_stratified_PG_loss_lower']
columns_tft = ['tft_total_rmse', 'tft_mixed_rmse', 'tft_upper_rmse', 'tft_lower_rmse', 'tft_total_PG_loss', 'tft_stratified_PG_loss_upper', 'tft_stratified_PG_loss_lower']
columns_obs = ['train_obs(days)','test_obs(days)']

total_columns = {'lstm': columns_lstm, 'pgrnn': columns_pgrnn, 'april': columns_april, 'pb': columns_pb, 'ealstm': columns_ealstm, 'transformer': columns_transformer, 'tft': columns_tft}

def check_if_skip(lake_id, group_id, seed):
    COLUMNS_USE = USE_FEATURES_COLUMNS_LAYER
    load_path = f"../../models/{len(COLUMNS_USE)}/seed={seed}/lstm/group_{group_id}/fine_tune/individual_train_on_obs/{lake_id}_lstm_fine_tune_train_on_obs"
    if not os.path.exists(load_path):
        return True
    ids = [lake_id]
    data_dir =  f'../../data/processed/'
    seq_length = 364
    win_shift = 364
    n_features = len(COLUMNS_USE) + 4
    (_, _, _, _, tst_data, _)  = buildManyLakeDataByIds(ids, data_dir, seq_length, n_features, win_shift, use_obs = True, evaluate = True)
    batch_size = tst_data.shape[0]

    if batch_size == 0:
        # csv_file_path =  f"../../data/utils/pred_NoObs_cluster_{group_id}.csv"
        # with open(csv_file_path, mode='a', newline='') as file:
        #     writer = csv.writer(file)
        #     new_row = [lake_id]
        #     writer.writerow(new_row)
        print("batch size:", batch_size)
        return True

    return False

def predict_and_write(group_id, model_index):
    seed = random_seed[model_index - 1]
    ids = pd.read_csv(f'../../data/utils/groups/vol_area/new_clusters/cluster_{group_id}.csv')
    ids = ids['nhdhr_id'].to_list()
    save_path_main = f'../../results/data/seed={seed}/'
    csv_name = f'cluster_{group_id}.csv'
    save_path = os.path.join(save_path_main, csv_name)
    if not os.path.exists(save_path_main): 
        os.makedirs(save_path_main)

    results = []
    columns = ['lake_id'] + columns_lstm + columns_pgrnn + columns_april + columns_pb + columns_ealstm + columns_transformer + columns_tft + columns_obs
    for count, lake_id in enumerate(ids):
        print("Lake id:", lake_id)
        if_skip = check_if_skip(lake_id, group_id, seed)
        if not if_skip:
            # reuslts of different models
            print("----------Lake: ", lake_id)
            results_lstm, obs_number = get_result("lstm", group_id, lake_id, seed, model_index)
            results_pgrnn, _ = get_result("pgrnn", group_id, lake_id, seed, model_index)
            results_april, _ = get_result("april", group_id, lake_id, seed, model_index)
            results_pb, _ = get_result("pb", group_id, lake_id, seed, model_index)
            results_ealstm, _ = get_result("ealstm", group_id, lake_id, seed, model_index)
            results_transformer, _ = get_result("transformer", group_id, lake_id, seed, model_index)
            results_tft = get_tft_result(group_id, lake_id, seed, model_index)
            print("results_tft:", results_tft)

            results_lstm = list(results_lstm)
            results_pgrnn = list(results_pgrnn)
            results_april = list(results_april)
            results_pb = list(results_pb)
            results_ealstm = list(results_ealstm)
            results_transformer = list(results_transformer)
            obs_number = list(obs_number)
            results_tft = list(results_tft)

            print("len results_tft:", len(results_tft))

            row = [lake_id] + results_lstm + results_pgrnn + results_april + results_pb + results_ealstm + results_transformer + results_tft + obs_number
            results.append(row)
        else:
            print(f"Skip lake: {lake_id}")

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

def predict_and_write_new(time, group_id, model_index, doc_sh):
    seed = random_seed[model_index - 1]
    ids = pd.read_csv(f'../../data/utils/groups/vol_area/removed/cluster_{group_id}.csv')
    ids = ids['nhdhr_id'].to_list()
    save_path_main = f'../../results/{time}/data/seed={seed}/'
    csv_name = f'cluster_{group_id}.csv'
    save_path = os.path.join(save_path_main, csv_name)
    if not os.path.exists(save_path_main): 
        os.makedirs(save_path_main)
    results = []
    columns = ['lake_id'] + columns_lstm + columns_pgrnn + columns_april + columns_pb + columns_ealstm + columns_transformer + columns_tft + columns_obs

    avg_results = {'lstm': [], 'pgrnn': [], 'april': [], 'pb': [], 'ealstm': [], 'transformer': [], 'tft': []}

    for count, lake_id in enumerate(ids):
        print("Lake id:", lake_id)
        if_skip = check_if_skip(lake_id, group_id, seed)
        if not if_skip:
            # reuslts of different models
            print("----------Lake: ", lake_id)
            results_lstm, obs_number = get_result("lstm", group_id, lake_id, seed, model_index, doc_sh)
            results_pgrnn, _ = get_result("pgrnn", group_id, lake_id, seed, model_index, doc_sh)
            results_april, _ = get_result("april", group_id, lake_id, seed, model_index, doc_sh)
            results_pb, _ = get_result("pb", group_id, lake_id, seed, model_index, doc_sh)
            results_ealstm, _ = get_result("ealstm", group_id, lake_id, seed, model_index, doc_sh)
            results_transformer, _ = get_result("transformer", group_id, lake_id, seed, model_index, doc_sh)
            results_tft = get_tft_result(group_id, lake_id, seed, model_index, doc_sh)
            print("results_tft:", results_tft)

            results_lstm = list(results_lstm)
            results_pgrnn = list(results_pgrnn)
            results_april = list(results_april)
            results_pb = list(results_pb)
            results_ealstm = list(results_ealstm)
            results_transformer = list(results_transformer)
            obs_number = list(obs_number)
            results_tft = list(results_tft)

            print("len results_tft:", len(results_tft))

            row = [lake_id] + results_lstm + results_pgrnn + results_april + results_pb + results_ealstm + results_transformer + results_tft + obs_number
            results.append(row)
        else:
            print(f"Skip lake: {lake_id}")

    df = pd.DataFrame(results, columns=columns)
    df.to_csv(save_path, index=False)
    print(f"Results saved to {save_path}")

    for model, values in avg_results.items():
        avg_results[model] = df[total_columns[model][1:]].mean()
    return avg_results



def get_result(model_type, group_id, lake_id, seed, model_index, doc_sh):
    use_gpu = False
    COLUMNS_USE = USE_FEATURES_COLUMNS_LAYER
    seq_length = 364
    win_shift = 364
    begin_loss_ind = 0
    n_features = len(COLUMNS_USE) + 4
    cluster_id = group_id

    if model_type == 'lstm':
        load_path = f"../../models/{len(COLUMNS_USE)}/seed={seed}/lstm/group_{cluster_id}/fine_tune/individual_train_on_obs/{lake_id}_lstm_fine_tune_train_on_obs"
    elif model_type == 'pgrnn':
        load_path = f"../../models/{len(COLUMNS_USE)}/seed={seed}/pgrnn/group_{cluster_id}/fine_tune/individual_train_on_obs/{lake_id}_pgrnn_fine_tune_train_on_obs"
    elif model_type == 'april':
        load_path = f"../../models/{len(COLUMNS_USE)}/seed={seed}/pgrnn/group_{cluster_id}/fine_tune/individual_train_on_obs_extend/{lake_id}_pgrnn_fine_tune_train_on_obs_12k"
    elif model_type == 'ealstm':
        load_path = f"../../models/{len(COLUMNS_USE)}/seed={seed}/ea_lstm/group_{cluster_id}/fine_tune/lambda=0/individual_train_on_obs/{lake_id}_ealstm_index_{model_index}_fine_tune_train_on_obs"
    elif model_type == 'transformer':
        load_path = f"../../models/{len(COLUMNS_USE)}/seed={seed}/transformer/group_{cluster_id}/fine_tune/lambda=0/individual_train_on_obs/{lake_id}_transformer_index_{model_index}_fine_tune_train_on_obs"
    else: # PB model 
        load_path = f"../../models/{len(COLUMNS_USE)}/seed={seed}/lstm/group_{cluster_id}/fine_tune/individual_train_on_obs/{lake_id}_lstm_fine_tune_train_on_obs"
        print("pb model")
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Model file {load_path} does not exist.")
    
    FLUX_START = n_features
    data_dir =  f'../../data/processed_extend/seed={seed}/'
    ids = [lake_id]
    (trn_data, _, val_data, _, tst_data, tst_dates)  = buildManyLakeDataByIds(ids, data_dir, seq_length, n_features, win_shift, use_obs = True, evaluate = True)
    print("dates start:", tst_dates[::2,0])
    print("dates end:", tst_dates[::2,-1])

    trn_data = torch.cat((trn_data, val_data), dim=0)

    obs_data_train = get_obs_number(trn_data)
    obs_data_test = get_obs_number(tst_data)

    batch_size = tst_data.shape[0]
    if batch_size == 0:
        csv_file_path =  f"../../data/utils/pred_NoObs_cluster_{group_id}.csv"
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            new_row = [lake_id]
            writer.writerow(new_row)
        return
    if model_type == 'transformer':
        transformer_cfg = TransformerConfig()
        num_heads = transformer_cfg.num_heads
        num_layers = transformer_cfg.num_layers
        hidden_size = transformer_cfg.hidden_size
        model = TransformerModel(input_size=n_features, num_heads=num_heads, num_layers=num_layers, hidden_size=hidden_size)
    elif model_type =='ealstm':
        n_hidden = 50
        print("n_hidden:", n_hidden)
        input_size_dyn = 35
        input_size_stat = 10
        model = MyEALSTM(input_size_dyn=input_size_dyn,
                        input_size_stat=input_size_stat,
                        hidden_size=n_hidden,
                        initial_forget_bias= 5,
                        dropout= 0.4,
                        concat_static=False,
                        no_static=False)
    else:
        n_hidden = torch.load(load_path)['state_dict']['out.weight'].shape[1]
        print("n_hidden:", n_hidden)
        model = MyLSTM(n_features, n_hidden, batch_size, use_gpu)



    pretrain_dict = torch.load(load_path)['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    model.load_state_dict(pretrain_dict)

    #things needed to predict test data
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    testloader = torch.utils.data.DataLoader(tst_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    print(tst_data.shape)
    print("--------")
    model.eval()
    with torch.no_grad():
        avg_mse = 0
        avg_epi_mse = 0
        avg_hypo_mse = 0
        total_DOC_conservation_loss = 0
        stratified_DOC_conservation_loss_upper = 0
        stratified_DOC_conservation_loss_lower = 0

        PB_total_loss = 0
        PB_stratified_loss_upper = 0
        PB_stratified_loss_lower = 0
        ct = 0
        for m, data in enumerate(testloader, 0):
            #now for mendota data
            #this loop is dated, there is now only one item in testloader
            #parse data into inputs and targets

            inputs = data[:,:,:n_features].float()
            sim_targets = data[:,:,-2].float()
            obs_targets = data[:,:,-1].float()

            Flux_data = data[:,:,FLUX_START:-2].float()

            obs_targets = obs_targets[:, begin_loss_ind:]
            nan_counts_targets = torch.isnan(obs_targets).sum(dim=1)
            tmp_dates = tst_dates[:, begin_loss_ind:]
            depths = inputs[:,:,0]
            if model_type == 'transformer':
                outputs = model(inputs)
            elif model_type == 'ealstm':
                dynamic_inputs = torch.cat((inputs[:, :, :29], inputs[:, :, 39:]), dim=2)
                static_inputs = inputs[:, 0, 29:39]
                outputs = model(dynamic_inputs, static_inputs)[0]
            else:
                h_state = None
                model.hidden = model.init_hidden(batch_size=inputs.size()[0])
                outputs, h_state = model(inputs, h_state)
            
            pred = outputs.view(outputs.size()[0],-1)
            pred = pred[:, begin_loss_ind:]
                        
            pb_pred = sim_targets[:, begin_loss_ind:]
            Volume = Flux_data[:,:,0]
            Volume[1::2,:] = Flux_data[1::2,:,1]
            ##################################################### Calculate MSE ##############################################################
            #calculate error
            loss_indices = np.array(np.isfinite(obs_targets.cpu()), dtype='bool_')
            loss_sim_indices = np.array(np.isfinite(sim_targets.cpu()), dtype='bool_')

            inputs = inputs[:, begin_loss_ind:, :]
            depths = depths[:, begin_loss_ind:]
            mse = mse_criterion(pred[loss_indices], obs_targets[loss_indices])
            pb_mse = mse_criterion(pb_pred[loss_indices], obs_targets[loss_indices])

            mae = mae_criterion(pred[loss_indices], obs_targets[loss_indices])
            pb_mae = mae_criterion(pb_pred[loss_indices], obs_targets[loss_indices])
            if mse > 0: #obsolete i think
                ct += 1
            
            upper_indices = np.arange(0, obs_targets.shape[0], 2)  # even epi
            lower_indices = np.arange(1, obs_targets.shape[0], 2)  # odd hypo
            mixed_mask = inputs[:, :, -6].cpu().numpy()
            total_layer_mask = (mixed_mask == 1)
            true_upper_layer_mask = (mixed_mask == 0)
            # mixed mse
            mixed_layer_indices = loss_indices * total_layer_mask
            mixed_layer_indices[1::2,:] = False
            mixed_mse = mse_criterion(pred[mixed_layer_indices], obs_targets[mixed_layer_indices])
            pb_mixed_mse = mse_criterion(pb_pred[mixed_layer_indices], obs_targets[mixed_layer_indices])

            mixed_mae = mae_criterion(pred[mixed_layer_indices], obs_targets[mixed_layer_indices])
            pb_mixed_mae = mae_criterion(pb_pred[mixed_layer_indices], obs_targets[mixed_layer_indices])

            # epi mse
            upper_loss_indices = loss_indices * true_upper_layer_mask
            upper_loss_indices[1::2,:] = False
            upper_mse = mse_criterion(pred[upper_loss_indices], obs_targets[upper_loss_indices])
            pb_upper_mse = mse_criterion(pb_pred[upper_loss_indices], obs_targets[upper_loss_indices])

            upper_mae = mae_criterion(pred[upper_loss_indices], obs_targets[upper_loss_indices])
            pb_upper_mae = mae_criterion(pb_pred[upper_loss_indices], obs_targets[upper_loss_indices])
        
            # hypo mse
            lower_loss_indices = loss_indices * true_upper_layer_mask
            lower_loss_indices[0::2,:] = False
            lower_mse = mse_criterion(pred[lower_loss_indices], obs_targets[lower_loss_indices])
            pb_lower_mse = mse_criterion(pb_pred[lower_loss_indices], obs_targets[lower_loss_indices])

            lower_mae = mae_criterion(pred[lower_loss_indices], obs_targets[lower_loss_indices])
            pb_lower_mae = mae_criterion(pb_pred[lower_loss_indices], obs_targets[lower_loss_indices])

            PB_pred = data[:,:,-2].float().unsqueeze(-1)
            for index in range(0,tst_data.shape[0], 2):
                # unsup_inputs.shape[0] == 2 !!, so there will only be one round 
                total_DOC_conservation_loss += (calculate_total_DOC_conservation_loss(Flux_data[index:index+2,:,:], outputs[index:index+2,:,:], doc_sh, 1, use_gpu) / (tst_data.shape[0]/2))
                lstm_uppser, lstm_lower = calculate_stratified_DOC_conservation_loss(Flux_data[index:index+2,:,:], outputs[index:index+2,:,:], doc_sh, 1, use_gpu)
                stratified_DOC_conservation_loss_upper += lstm_uppser/(tst_data.shape[0]/2)
                stratified_DOC_conservation_loss_lower += lstm_lower/(tst_data.shape[0]/2)

                # print("====================================== PB start================================")
                PB_total_loss += (calculate_total_DOC_conservation_loss(Flux_data[index:index+2,:,:], PB_pred[index:index+2,:,:], doc_sh, 1, use_gpu) / (tst_data.shape[0]/2))
                pb_upper, pb_lower = calculate_stratified_DOC_conservation_loss(Flux_data[index:index+2,:,:], PB_pred[index:index+2,:,:], doc_sh, 1, use_gpu)
                PB_stratified_loss_upper += pb_upper/(tst_data.shape[0]/2)
                PB_stratified_loss_lower += pb_lower/(tst_data.shape[0]/2)
    pb_total_rmse = (pb_mse)**0.5
    pb_mixed_rmse = (pb_mixed_mse)**0.5
    pb_upper_rmse = (pb_upper_mse)**0.5
    pb_lower_rmse = (pb_lower_mse)**0.5

    lstm_total_rmse = (mse)**0.5
    lstm_mixed_rmse = (mixed_mse )**0.5
    lstm_upper_rmse = (upper_mse)**0.5
    lstm_lower_rmse = (lower_mse )**0.5
    assert ct == 1, f"Error count should be 1 not {ct}"
    print("model name:", load_path)
    print("model type:", model_type)
    print("Lake ID:", lake_id)
    results_pbmodel = [pb_total_rmse, pb_mixed_rmse, pb_upper_rmse, pb_lower_rmse, PB_total_loss, PB_stratified_loss_upper, PB_stratified_loss_lower]
    results_ai = [lstm_total_rmse, lstm_mixed_rmse, lstm_upper_rmse, lstm_lower_rmse, total_DOC_conservation_loss, stratified_DOC_conservation_loss_upper, stratified_DOC_conservation_loss_lower]

    obs_data = [obs_data_train, obs_data_test]

    results_ai = [val.item() if isinstance(val, torch.Tensor) else val for val in results_ai]
    results_pbmodel = [val.item() if isinstance(val, torch.Tensor) else val for val in results_pbmodel]
    obs_data = [val.item() if isinstance(val, torch.Tensor) else val for val in obs_data]

    if model_type == 'pb':
        return results_pbmodel, obs_data
    else:
        return results_ai, obs_data

def get_tft_result(group_id, lake_id, seed, model_index, doc_sh):
    use_gpu = False
    COLUMNS_USE = USE_FEATURES_COLUMNS_LAYER
    seq_length = 364
    win_shift = 364
    begin_loss_ind = 0
    n_features = len(COLUMNS_USE) + 4
    FLUX_START = n_features
    data_dir =  f'../../data/processed_extend/seed={seed}/'
    ids = [lake_id]
    (trn_data, _, val_data, _, tst_data, tst_dates)  = buildManyLakeDataByIds_ForTFT(group_id, ids, seed, data_dir, seq_length, n_features, win_shift, use_obs = True, evaluate = True)
    print("dates start:", tst_dates[::2,0])
    print("dates end:", tst_dates[::2,-1])

    trn_data = torch.cat((trn_data, val_data), dim=0)
    obs_data_train = get_obs_number(trn_data)
    obs_data_test = get_obs_number(tst_data)
    batch_size = tst_data.shape[0]
    if batch_size == 0:
        csv_file_path =  f"../../data/utils/pred_NoObs_cluster_{group_id}.csv"
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            new_row = [lake_id]
            writer.writerow(new_row)
        return

    #things needed to predict test data
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    testloader = torch.utils.data.DataLoader(tst_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    with torch.no_grad():
        avg_mse = 0
        avg_epi_mse = 0
        avg_hypo_mse = 0
        total_DOC_conservation_loss = 0
        stratified_DOC_conservation_loss_upper = 0
        stratified_DOC_conservation_loss_lower = 0

        PB_total_loss = 0
        PB_stratified_loss_upper = 0
        PB_stratified_loss_lower = 0
        ct = 0
        for m, data in enumerate(testloader, 0):
            #now for mendota data
            #this loop is dated, there is now only one item in testloader
            #parse data into inputs and targets

            inputs = data[:,:,:n_features].float()
            sim_targets = data[:,:,-2].float()
            obs_targets = data[:,:,-1].float()

            Flux_data = data[:,:,FLUX_START:-2].float()

            obs_targets = obs_targets[:, begin_loss_ind:]
            nan_counts_targets = torch.isnan(obs_targets).sum(dim=1)
            tmp_dates = tst_dates[:, begin_loss_ind:]
            depths = inputs[:,:,0]


                        
            pb_pred = sim_targets[:, begin_loss_ind:]
            Volume = Flux_data[:,:,0]
            Volume[1::2,:] = Flux_data[1::2,:,1]
            ##################################################### Calculate MSE ##############################################################
            #calculate error
            loss_indices = np.array(np.isfinite(obs_targets.cpu()), dtype='bool_')
            loss_sim_indices = np.array(np.isfinite(sim_targets.cpu()), dtype='bool_')

            inputs = inputs[:, begin_loss_ind:, :]
            depths = depths[:, begin_loss_ind:]
            pb_mse = mse_criterion(pb_pred[loss_indices], obs_targets[loss_indices])
            pb_mae = mae_criterion(pb_pred[loss_indices], obs_targets[loss_indices])
            if pb_mse > 0: #obsolete i think
                ct += 1
            
            upper_indices = np.arange(0, obs_targets.shape[0], 2)  # even epi
            lower_indices = np.arange(1, obs_targets.shape[0], 2)  # odd hypo
            mixed_mask = inputs[:, :, -6].cpu().numpy()
            total_layer_mask = (mixed_mask == 1)
            true_upper_layer_mask = (mixed_mask == 0)
            # mixed mse
            mixed_layer_indices = loss_indices * total_layer_mask
            mixed_layer_indices[1::2,:] = False
            pb_mixed_mse = mse_criterion(pb_pred[mixed_layer_indices], obs_targets[mixed_layer_indices])

            pb_mixed_mae = mae_criterion(pb_pred[mixed_layer_indices], obs_targets[mixed_layer_indices])

            # epi mse
            upper_loss_indices = loss_indices * true_upper_layer_mask
            upper_loss_indices[1::2,:] = False
            pb_upper_mse = mse_criterion(pb_pred[upper_loss_indices], obs_targets[upper_loss_indices])

            pb_upper_mae = mae_criterion(pb_pred[upper_loss_indices], obs_targets[upper_loss_indices])
        
            # hypo mse
            lower_loss_indices = loss_indices * true_upper_layer_mask
            lower_loss_indices[0::2,:] = False
            pb_lower_mse = mse_criterion(pb_pred[lower_loss_indices], obs_targets[lower_loss_indices])

            pb_lower_mae = mae_criterion(pb_pred[lower_loss_indices], obs_targets[lower_loss_indices])

            PB_pred = data[:,:,-2].float().unsqueeze(-1)
            for index in range(0,tst_data.shape[0], 2):
                # print("====================================== PB start================================")
                PB_total_loss += (calculate_total_DOC_conservation_loss(Flux_data[index:index+2,:,:], PB_pred[index:index+2,:,:], doc_sh, 1, use_gpu) / (tst_data.shape[0]/2))
                pb_upper, pb_lower = calculate_stratified_DOC_conservation_loss(Flux_data[index:index+2,:,:], PB_pred[index:index+2,:,:], doc_sh, 1, use_gpu)
                PB_stratified_loss_upper += pb_upper/(tst_data.shape[0]/2)
                PB_stratified_loss_lower += pb_lower/(tst_data.shape[0]/2)
    pb_total_rmse = (pb_mse)**0.5
    pb_mixed_rmse = (pb_mixed_mse)**0.5
    pb_upper_rmse = (pb_upper_mse)**0.5
    pb_lower_rmse = (pb_lower_mse)**0.5


    assert ct == 1, f"Error count should be 1 not {ct}"

    print("Lake ID:", lake_id)
    results_pbmodel = [pb_total_rmse, pb_mixed_rmse, pb_upper_rmse, pb_lower_rmse, PB_total_loss, PB_stratified_loss_upper, PB_stratified_loss_lower]

    obs_data = [obs_data_train, obs_data_test]

    results_pbmodel = [val.item() if isinstance(val, torch.Tensor) else val for val in results_pbmodel]
    obs_data = [val.item() if isinstance(val, torch.Tensor) else val for val in obs_data]
    print(f"results_tft lake:{lake_id} is:", results_pbmodel)

    return results_pbmodel
