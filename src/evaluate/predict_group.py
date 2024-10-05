import pandas as pd
import numpy as np
import sys
import os
sys.path.append('../../data')
sys.path.append('../data')
sys.path.append('../../models')
sys.path.append('../models')
sys.path.append('../../utils')
sys.path.append('../utils')
from lakePreprocess import DROP_COLUMNS, USE_FEATURES_COLUMNS, FLUX_COLUMNS, USE_FEATURES_COLUMNS_LAYER
from pytorch_data_operations import buildManyLakeDataByIds, calculate_total_DOC_conservation_loss, calculate_stratified_DOC_conservation_loss
import csv
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal_
from sklearn.ensemble import GradientBoostingRegressor
from scipy.stats import spearmanr
from joblib import dump, load
import re
import matplotlib.pyplot as plt
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from matplotlib.legend_handler import HandlerLine2D

from train_utils import TransformerModel, TransformerConfig

class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, use_gpu):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.lstm = nn.LSTM(input_size = input_size, hidden_size=hidden_size, batch_first=True) 
        self.out = nn.Linear(hidden_size, 1)
        self.hidden = self.init_hidden()
        # self.w_upper_to_lower = []
        # self.w_lower_to_upper = []           

    def init_hidden(self, batch_size=0):
        # initialize both hidden layers
        if batch_size == 0:
            batch_size = self.batch_size
        ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
                xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
        if self.use_gpu:
            item0 = ret[0].cuda(non_blocking=True)
            item1 = ret[1].cuda(non_blocking=True)
            ret = (item0,item1)
        return ret
    
    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        x = x.float()
        x, hidden = self.lstm(x, self.hidden)
        self.hidden = hidden
        x = self.out(x)
        x = F.relu(x)
        return x, hidden
    
def get_obs_number(input_data):
    obs_data = input_data[:, :, -1].float()
    valid_number = torch.sum(~torch.isnan(obs_data))
    return valid_number

def check_if_skip(lake_id, group_id):
    COLUMNS_USE = USE_FEATURES_COLUMNS_LAYER
    load_path = f"../../models/{len(COLUMNS_USE)}/lstm/group_{group_id}/fine_tune/individual_train_on_obs/{lake_id}_lstm_fine_tune_train_on_obs"
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
        csv_file_path =  f"../../data/utils/pred_NoObs_cluster_{group_id}.csv"
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            new_row = [lake_id]
            writer.writerow(new_row)
        return True

    return False


def test_lstm(lake_id, group_id, if_cluster_model):
    use_gpu = False
    COLUMNS_USE = USE_FEATURES_COLUMNS_LAYER
    seq_length = 364
    win_shift = 364
    begin_loss_ind = 0
    n_features = len(COLUMNS_USE) + 4
    cluster_id = group_id
    if not if_cluster_model:
        load_path = f"../../models/{len(COLUMNS_USE)}/lstm/group_{cluster_id}/fine_tune/individual_train_on_obs/{lake_id}_lstm_fine_tune_train_on_obs"
    else:
        load_path = f"../../models/{len(COLUMNS_USE)}/lstm/group_{cluster_id}/fine_tune/lambda=0/group_{cluster_id}_lstm_model_fine_tune_41_1_lambda-0"
    if not os.path.exists(load_path):
        return 

    layer_extended = 4 # Increasing the dimension representing the layer from 1 to 5.
    FLUX_START = n_features
    data_dir =  f'../../data/processed_extend/'
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

    n_hidden = torch.load(load_path)['state_dict']['out.weight'].shape[1]
    print("n_hidden:", n_hidden)
    # load model 1
    lstm_net = MyLSTM(n_features, n_hidden, batch_size, use_gpu)
    pretrain_dict = torch.load(load_path)['state_dict']
    model_dict = lstm_net.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    lstm_net.load_state_dict(pretrain_dict)

    #things needed to predict test data
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    testloader = torch.utils.data.DataLoader(tst_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    print(tst_data.shape)
    print("--------")
    lstm_net.eval()
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
            nan_counts_sim_targets = torch.isnan(sim_targets).sum(dim=1)
            print("sim_targets shape:", sim_targets.shape)
            print("nan in sim_targets:", nan_counts_sim_targets)
            tmp_dates = tst_dates[:, begin_loss_ind:]
            depths = inputs[:,:,0]

            #run model 1
            h_state = None
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            outputs, h_state = lstm_net(inputs, h_state)
            
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
            lower_loss_indices = loss_indices[lower_indices, :]
            lower_mse = mse_criterion(pred[lower_indices, :][lower_loss_indices], obs_targets[lower_indices, :][lower_loss_indices])
            pb_lower_mse = mse_criterion(pb_pred[lower_indices, :][lower_loss_indices], obs_targets[lower_indices, :][lower_loss_indices])

            lower_mae = mae_criterion(pred[lower_indices, :][lower_loss_indices], obs_targets[lower_indices, :][lower_loss_indices])
            pb_lower_mae = mae_criterion(pb_pred[lower_indices, :][lower_loss_indices], obs_targets[lower_indices, :][lower_loss_indices])

            ##################################################### Calculate PGRNN Loss #######################################################
            PB_pred = data[:,:,-2].float().unsqueeze(-1)
            for index in range(0,tst_data.shape[0], 2):
                # unsup_inputs.shape[0] == 2 !!, so there will only be one round 
                total_DOC_conservation_loss += (calculate_total_DOC_conservation_loss(Flux_data[index:index+2,:,:], outputs[index:index+2,:,:], 0, 1, use_gpu) / (tst_data.shape[0]/2))
                lstm_uppser, lstm_lower = calculate_stratified_DOC_conservation_loss(Flux_data[index:index+2,:,:], outputs[index:index+2,:,:], 0, 1, use_gpu)
                stratified_DOC_conservation_loss_upper += lstm_uppser/(tst_data.shape[0]/2)
                stratified_DOC_conservation_loss_lower += lstm_lower/(tst_data.shape[0]/2)

                # print("====================================== PB start================================")
                PB_total_loss += (calculate_total_DOC_conservation_loss(Flux_data[index:index+2,:,:], PB_pred[index:index+2,:,:], 0, 1, use_gpu) / (tst_data.shape[0]/2))
                pb_upper, pb_lower = calculate_stratified_DOC_conservation_loss(Flux_data[index:index+2,:,:], PB_pred[index:index+2,:,:], 0, 1, use_gpu)
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
    print("==================RMSE RESULT==================")
    print("------------------PB model--------------------")
    print("PB Total RMSE:", pb_total_rmse)
    print("PB Mixed RMSE:", pb_mixed_rmse)
    print("PB EPI RMSE:", pb_upper_rmse)
    print("PB HYPO RMSE:", pb_lower_rmse)
    print("PB Total Loss:", PB_total_loss)
    print("PB Stratified Loss Upper:", PB_stratified_loss_upper)
    print("PB Stratified Loss Lower:", PB_stratified_loss_lower)

    print("-----------------Ai model prediction----------")
    print("Total RMSE:", lstm_total_rmse)
    print("Mixed RMSE:", lstm_mixed_rmse)
    print("EPI RMSE:", lstm_upper_rmse)
    print("HYPO RMSE:", lstm_lower_rmse)
    print("PGRNN Total Loss:", total_DOC_conservation_loss)
    print("PGRNN Stratified Loss Lower:", stratified_DOC_conservation_loss_upper)
    print("PGRNN Stratified Loss Lower:", stratified_DOC_conservation_loss_lower)
    # print("==================MAE RESULT==================")
    # # print("------------------PB model--------------------")
    # # print("PB Total MAE:", (pb_mae ))
    # # print("PB Mixed MAE:", (pb_mixed_mae ))
    # # print("PB EPI MAE:", (pb_upper_mae))
    # # print("PB HYPO MAE:", (pb_lower_mae ))

    # print("-----------------Ai model prediction----------")
    # print("Total MAE:", (mae))
    # print("Mixed MAE:", (mixed_mae ))
    # print("EPI MAE:", (upper_mae))
    # print("HYPO MAE:", (lower_mae ))

    print("Lake ID:", lake_id)
    results_pbmodel = [pb_total_rmse, pb_mixed_rmse, pb_upper_rmse, pb_lower_rmse, PB_total_loss, PB_stratified_loss_upper, PB_stratified_loss_lower]
    results_lstm = [lstm_total_rmse, lstm_mixed_rmse, lstm_upper_rmse, lstm_lower_rmse, total_DOC_conservation_loss, stratified_DOC_conservation_loss_upper, stratified_DOC_conservation_loss_lower]
    # print("results_lstm:", results_lstm)
    # print("results_pbmodel:", results_pbmodel)
    return results_lstm, results_pbmodel, [obs_data_train, obs_data_test]


def test_pgrnn(lake_id, group_id, if_cluster_model):
    use_gpu = False
    COLUMNS_USE = USE_FEATURES_COLUMNS_LAYER
    seq_length = 364
    win_shift = 364
    begin_loss_ind = 0
    n_features = len(COLUMNS_USE) + 4
    cluster_id = group_id
    if not if_cluster_model:
        load_path = f"../../models/{len(COLUMNS_USE)}/pgrnn/group_{cluster_id}/fine_tune/individual_train_on_obs/{lake_id}_pgrnn_fine_tune_train_on_obs"
    else:
        load_path = f"../../models/{len(COLUMNS_USE)}/pgrnn/group_{cluster_id}/fine_tune/lambda=1/group_{cluster_id}_pgrnn_model_fine_tune_41_1_lambda-1"
    if not os.path.exists(load_path):
        return 

    layer_extended = 4 # Increasing the dimension representing the layer from 1 to 5.
    FLUX_START = n_features
    data_dir =  f'../../data/processed_extend/'
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

    n_hidden = torch.load(load_path)['state_dict']['out.weight'].shape[1]
    print("n_hidden:", n_hidden)
    # load model 1
    lstm_net = MyLSTM(n_features, n_hidden, batch_size, use_gpu)
    pretrain_dict = torch.load(load_path)['state_dict']
    model_dict = lstm_net.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    lstm_net.load_state_dict(pretrain_dict)

    #things needed to predict test data
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    testloader = torch.utils.data.DataLoader(tst_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    print(tst_data.shape)
    print("--------")
    lstm_net.eval()
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

            #run model 1
            h_state = None
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            outputs, h_state = lstm_net(inputs, h_state)
            
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
            ####################################3
            # lower_loss_indices = loss_indices[lower_indices, :]
            # lower_mse = mse_criterion(pred[lower_indices, :][lower_loss_indices], obs_targets[lower_indices, :][lower_loss_indices])
            # pb_lower_mse = mse_criterion(pb_pred[lower_indices, :][lower_loss_indices], obs_targets[lower_indices, :][lower_loss_indices])

            # lower_mae = mae_criterion(pred[lower_indices, :][lower_loss_indices], obs_targets[lower_indices, :][lower_loss_indices])
            # pb_lower_mae = mae_criterion(pb_pred[lower_indices, :][lower_loss_indices], obs_targets[lower_indices, :][lower_loss_indices])

            ##################################################### Calculate PGRNN Loss #######################################################
            PB_pred = data[:,:,-2].float().unsqueeze(-1)
            for index in range(0,tst_data.shape[0], 2):
                # unsup_inputs.shape[0] == 2 !!, so there will only be one round 
                total_DOC_conservation_loss += (calculate_total_DOC_conservation_loss(Flux_data[index:index+2,:,:], outputs[index:index+2,:,:], 0, 1, use_gpu) / (tst_data.shape[0]/2))
                lstm_uppser, lstm_lower = calculate_stratified_DOC_conservation_loss(Flux_data[index:index+2,:,:], outputs[index:index+2,:,:], 0, 1, use_gpu)
                stratified_DOC_conservation_loss_upper += lstm_uppser/(tst_data.shape[0]/2)
                stratified_DOC_conservation_loss_lower += lstm_lower/(tst_data.shape[0]/2)

                # print("====================================== PB start================================")
                PB_total_loss += (calculate_total_DOC_conservation_loss(Flux_data[index:index+2,:,:], PB_pred[index:index+2,:,:], 0, 1, use_gpu) / (tst_data.shape[0]/2))
                pb_upper, pb_lower = calculate_stratified_DOC_conservation_loss(Flux_data[index:index+2,:,:], PB_pred[index:index+2,:,:], 0, 1, use_gpu)
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
    print("==================RMSE RESULT==================")
    print("------------------PB model--------------------")
    print("PB Total RMSE:", pb_total_rmse)
    print("PB Mixed RMSE:", pb_mixed_rmse)
    print("PB EPI RMSE:", pb_upper_rmse)
    print("PB HYPO RMSE:", pb_lower_rmse)
    print("PB Total Loss:", PB_total_loss)
    print("PB Stratified Loss Upper:", PB_stratified_loss_upper)
    print("PB Stratified Loss Lower:", PB_stratified_loss_lower)

    print("-----------------Ai model prediction----------")
    print("Total RMSE:", lstm_total_rmse)
    print("Mixed RMSE:", lstm_mixed_rmse)
    print("EPI RMSE:", lstm_upper_rmse)
    print("HYPO RMSE:", lstm_lower_rmse)
    print("PGRNN Total Loss:", total_DOC_conservation_loss)
    print("PGRNN Stratified Loss Lower:", stratified_DOC_conservation_loss_upper)
    print("PGRNN Stratified Loss Lower:", stratified_DOC_conservation_loss_lower)
    # print("==================MAE RESULT==================")
    # # print("------------------PB model--------------------")
    # # print("PB Total MAE:", (pb_mae ))
    # # print("PB Mixed MAE:", (pb_mixed_mae ))
    # # print("PB EPI MAE:", (pb_upper_mae))
    # # print("PB HYPO MAE:", (pb_lower_mae ))

    # print("-----------------Ai model prediction----------")
    # print("Total MAE:", (mae))
    # print("Mixed MAE:", (mixed_mae ))
    # print("EPI MAE:", (upper_mae))
    # print("HYPO MAE:", (lower_mae ))

    print("Lake ID:", lake_id)
    results_pbmodel = [pb_total_rmse, pb_mixed_rmse, pb_upper_rmse, pb_lower_rmse, PB_total_loss, PB_stratified_loss_upper, PB_stratified_loss_lower]
    results_lstm = [lstm_total_rmse, lstm_mixed_rmse, lstm_upper_rmse, lstm_lower_rmse, total_DOC_conservation_loss, stratified_DOC_conservation_loss_upper, stratified_DOC_conservation_loss_lower]
    return results_lstm, results_pbmodel, [obs_data_train, obs_data_test]

def test_pgrnn_extend(lake_id, group_id, if_cluster_model):
    use_gpu = False
    COLUMNS_USE = USE_FEATURES_COLUMNS_LAYER
    seq_length = 364
    win_shift = 364
    begin_loss_ind = 0
    n_features = len(COLUMNS_USE) + 4
    cluster_id = group_id
    if not if_cluster_model:
        load_path = f"../../models/{len(COLUMNS_USE)}/pgrnn/group_{cluster_id}/fine_tune/individual_train_on_obs_extend/{lake_id}_pgrnn_fine_tune_train_on_obs"
    else:
        load_path = f"../../models/{len(COLUMNS_USE)}/pgrnn/group_{cluster_id}/fine_tune/lambda=1/group_{cluster_id}_pgrnn_model_fine_tune_41_1_lambda-1"
    if not os.path.exists(load_path):
        return 

    layer_extended = 4 # Increasing the dimension representing the layer from 1 to 5.
    FLUX_START = n_features
    data_dir =  f'../../data/processed_extend/'
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

    n_hidden = torch.load(load_path)['state_dict']['out.weight'].shape[1]
    print("n_hidden:", n_hidden)
    # load model 1
    lstm_net = MyLSTM(n_features, n_hidden, batch_size, use_gpu)
    pretrain_dict = torch.load(load_path)['state_dict']
    model_dict = lstm_net.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    lstm_net.load_state_dict(pretrain_dict)

    #things needed to predict test data
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    testloader = torch.utils.data.DataLoader(tst_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    print(tst_data.shape)
    print("--------")
    lstm_net.eval()
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

            #run model 1
            h_state = None
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            outputs, h_state = lstm_net(inputs, h_state)
            
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
            ####################################3
            # lower_loss_indices = loss_indices[lower_indices, :]
            # lower_mse = mse_criterion(pred[lower_indices, :][lower_loss_indices], obs_targets[lower_indices, :][lower_loss_indices])
            # pb_lower_mse = mse_criterion(pb_pred[lower_indices, :][lower_loss_indices], obs_targets[lower_indices, :][lower_loss_indices])

            # lower_mae = mae_criterion(pred[lower_indices, :][lower_loss_indices], obs_targets[lower_indices, :][lower_loss_indices])
            # pb_lower_mae = mae_criterion(pb_pred[lower_indices, :][lower_loss_indices], obs_targets[lower_indices, :][lower_loss_indices])

            ##################################################### Calculate PGRNN Loss #######################################################
            PB_pred = data[:,:,-2].float().unsqueeze(-1)
            for index in range(0,tst_data.shape[0], 2):
                # unsup_inputs.shape[0] == 2 !!, so there will only be one round 
                total_DOC_conservation_loss += (calculate_total_DOC_conservation_loss(Flux_data[index:index+2,:,:], outputs[index:index+2,:,:], 0, 1, use_gpu) / (tst_data.shape[0]/2))
                lstm_uppser, lstm_lower = calculate_stratified_DOC_conservation_loss(Flux_data[index:index+2,:,:], outputs[index:index+2,:,:], 0, 1, use_gpu)
                stratified_DOC_conservation_loss_upper += lstm_uppser/(tst_data.shape[0]/2)
                stratified_DOC_conservation_loss_lower += lstm_lower/(tst_data.shape[0]/2)

                # print("====================================== PB start================================")
                PB_total_loss += (calculate_total_DOC_conservation_loss(Flux_data[index:index+2,:,:], PB_pred[index:index+2,:,:], 0, 1, use_gpu) / (tst_data.shape[0]/2))
                pb_upper, pb_lower = calculate_stratified_DOC_conservation_loss(Flux_data[index:index+2,:,:], PB_pred[index:index+2,:,:], 0, 1, use_gpu)
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
    print("==================RMSE RESULT==================")
    print("------------------PB model--------------------")
    print("PB Total RMSE:", pb_total_rmse)
    print("PB Mixed RMSE:", pb_mixed_rmse)
    print("PB EPI RMSE:", pb_upper_rmse)
    print("PB HYPO RMSE:", pb_lower_rmse)
    print("PB Total Loss:", PB_total_loss)
    print("PB Stratified Loss Upper:", PB_stratified_loss_upper)
    print("PB Stratified Loss Lower:", PB_stratified_loss_lower)

    print("-----------------Ai model prediction----------")
    print("Total RMSE:", lstm_total_rmse)
    print("Mixed RMSE:", lstm_mixed_rmse)
    print("EPI RMSE:", lstm_upper_rmse)
    print("HYPO RMSE:", lstm_lower_rmse)
    print("PGRNN Total Loss:", total_DOC_conservation_loss)
    print("PGRNN Stratified Loss Lower:", stratified_DOC_conservation_loss_upper)
    print("PGRNN Stratified Loss Lower:", stratified_DOC_conservation_loss_lower)
    # print("==================MAE RESULT==================")
    # # print("------------------PB model--------------------")
    # # print("PB Total MAE:", (pb_mae ))
    # # print("PB Mixed MAE:", (pb_mixed_mae ))
    # # print("PB EPI MAE:", (pb_upper_mae))
    # # print("PB HYPO MAE:", (pb_lower_mae ))

    # print("-----------------Ai model prediction----------")
    # print("Total MAE:", (mae))
    # print("Mixed MAE:", (mixed_mae ))
    # print("EPI MAE:", (upper_mae))
    # print("HYPO MAE:", (lower_mae ))

    print("Lake ID:", lake_id)
    results_pbmodel = [pb_total_rmse, pb_mixed_rmse, pb_upper_rmse, pb_lower_rmse, PB_total_loss, PB_stratified_loss_upper, PB_stratified_loss_lower]
    results_lstm = [lstm_total_rmse, lstm_mixed_rmse, lstm_upper_rmse, lstm_lower_rmse, total_DOC_conservation_loss, stratified_DOC_conservation_loss_upper, stratified_DOC_conservation_loss_lower]
    return results_lstm, results_pbmodel, [obs_data_train, obs_data_test]

def draw_pic(save_path, pic_model_name, Date_epi, Date_hypo, pred_2_epi, pred_2_hypo, obs_indices_epi, obs_data_epi, sim_indices_epi, sim_data_epi, obs_indices_hypo, obs_data_hypo, sim_indices_hypo, sim_data_hypo):
    plt.figure(figsize=(14, 5), dpi=100)

    colors_epi = {
        'pred_epi': '#6495ED',  # CornflowerBlue
        'pred_hypo': '#87CEFA',  # LightSkyBlue
        'sim_epi': '#3CB371',   # MediumSeaGreen
        'sim_hypo': '#66CDAA',   # MediumAquamarine
        'obs_epi': '#f296de',   # LightSalmon
        'obs_hypo': '#eb7e4b',   # LightSalmon4
    }

    #pre-trained
    # plt.plot(Date_epi, pred_1_epi, label=f'{model_status1}-model-{model_type}', color=colors_epi['pred2'])
    #fine-tune
    plt.plot(Date_hypo, pred_2_hypo, label=f'{pic_model_name} Hypo', color=colors_epi['pred_hypo'])  
    plt.plot(Date_epi, pred_2_epi, label=f'{pic_model_name} Epi', color=colors_epi['pred_epi'])  
    

    if len(obs_indices_epi) > 0: 
        plt.scatter(Date_epi[obs_indices_epi], obs_data_epi[obs_indices_epi], label='Observe Epi', color=colors_epi['obs_epi'], s=36, edgecolor='#FFFFFF')
   

    if len(obs_indices_hypo) > 0: 
        plt.scatter(Date_hypo[obs_indices_hypo], obs_data_hypo[obs_indices_hypo], label='Observe Hypo', color=colors_epi['obs_hypo'], s=36, edgecolor='#FFFFFF')


    if len(sim_indices_hypo) > 0: 
        plt.plot(Date_hypo[sim_indices_hypo], sim_data_hypo[sim_indices_hypo], label='Simulator Hypo', linestyle='--', color=colors_epi['sim_hypo'])


    if len(sim_indices_epi) > 0: 
        plt.plot(Date_epi[sim_indices_epi], sim_data_epi[sim_indices_epi], label='Simulator Epi', linestyle='--', color=colors_epi['sim_epi'])


    plt.title('Daily oxygen concentrations', fontsize=14)
    plt.xlabel('Time', fontsize=12)
    plt.ylabel('DO concentration', fontsize=12)
    plt.legend()

    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m'))
    plt.gca().xaxis.set_major_locator(mdates.MonthLocator())
    plt.gcf().autofmt_xdate()


    plt.grid(True, which='both', linestyle='--', linewidth=0.5)


    plt.tight_layout()
    # plt.show()

    plt.savefig(save_path)
    plt.close()


class CustomHandler(HandlerLine2D):
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        artists = super().create_artists(legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans)
        lw = orig_handle.get_linewidth() * 7  
        line = plt.Line2D([xdescent, width - xdescent], [height/2. - ydescent, height/2. - ydescent],
                          linestyle=orig_handle.get_linestyle(),
                          linewidth=lw, color=orig_handle.get_color(), alpha=0.3)
        artists.insert(0, line)
        
        return artists
    

def draw_pic_beauty(save_path, pic_model_name, Date_epi, Date_hypo, pred_2_epi, pred_2_hypo, pred_2_mixed, obs_indices_epi, obs_data_epi, sim_indices_epi, sim_data_epi, obs_indices_hypo, obs_data_hypo, sim_indices_hypo, sim_data_hypo, obs_indices_mixed, sim_data_mixed, sim_indices_mixed, pred_indices_mixed):
    plt.figure(figsize=(12, 4))
    # Updated color palette for clear distinction
    c_pred = '#1f77b4'  # Blue for NGCE predictions
    c_label = '#ff7f0e'  # Orange for Simulated DO concentrations
    c_obs_epi = '#2ca02c'  # Green for Obs DO conc. (epi)
    c_obs_hyp = '#d62728'  # Red for Obs DO conc. (hyp)
    c_fill_pred = '#8ea7d4'  # Lighter blue for prediction fill
    c_fill_label = '#ffd59e'  # Lighter orange for label fill
    c_obs_mixed = '#bebbbb' # Gray
    

    plt.style.use('seaborn-whitegrid')  # Use a clean and professional style
    FONT_SIZE1 = 18  # Slightly reduced for balance
    FONT_SIZE2 = 15
    FONT_SIZE_LEG = 15.5
    LINEWIDTH = 0.9 # More elegant line width
    LINEWIDTH2 = 0.9
    LINEWIDTH_LASER = 6.5

    #pre-trained
    # plt.plot(Date_epi, pred_1_epi, label=f'{model_status1}-model-{model_type}', color=colors_epi['pred2'])
    #fine-tune
    if pic_model_name == 'April' or pic_model_name == 'Pril':
        plt.plot(Date_epi, pred_2_epi, label= f"$\t{{{pic_model_name}}}$ (epi)", color=c_pred, linewidth=LINEWIDTH)  
        plt.plot(Date_hypo, pred_2_hypo, label= f"$\t{{{pic_model_name}}}$ (hyp)", color=c_pred, linewidth=LINEWIDTH, linestyle='--')  

        # laser line
        line_mixed_pred, = plt.plot(Date_epi, pred_2_mixed, label= f"$\t{{{pic_model_name}}}$ (total)", color=c_pred, linewidth=LINEWIDTH2) 
        plt.plot(Date_epi, pred_2_mixed, color=c_pred, linewidth=LINEWIDTH_LASER, alpha=0.3) 
    else:
        plt.plot(Date_epi, pred_2_epi, label=f"{pic_model_name} (epi)", color=c_pred, linewidth=LINEWIDTH)  
        plt.plot(Date_hypo, pred_2_hypo, label=f"{pic_model_name} (hyp)", color=c_pred, linewidth=LINEWIDTH, linestyle='--') 

        # laser line 
        line_mixed_pred, = plt.plot(Date_epi, pred_2_mixed, label=f"{pic_model_name} (total)", color=c_pred, linewidth=LINEWIDTH2)
        plt.plot(Date_epi, pred_2_mixed, color=c_pred, linewidth=LINEWIDTH_LASER, alpha=0.3)
    # Fill between Pred APRIL DO concentrations
    plt.fill_between(Date_hypo, pred_2_epi, pred_2_hypo, color=c_fill_pred, alpha=0.4)



    plt.plot(Date_epi, sim_data_epi, label='Process (epi)', color=c_label, alpha=0.8, linewidth=LINEWIDTH)

    plt.plot(Date_hypo, sim_data_hypo, label='Process (hyp)', linestyle='--', color=c_label, alpha=0.8, linewidth=LINEWIDTH)

    line_mixed_sim, = plt.plot(Date_epi, sim_data_mixed, label='Process (total)', color=c_label, alpha=0.8, linewidth=LINEWIDTH2)
    plt.plot(Date_epi, sim_data_mixed, color=c_label, alpha=0.3, linewidth=LINEWIDTH_LASER)

    # Fill between Simulated DO concentrations
    plt.fill_between(Date_epi, sim_data_epi[sim_indices_epi], sim_data_hypo[sim_indices_hypo], color=c_fill_label, alpha=0.3)



    if len(obs_indices_epi) > 0: 
        for size, alpha in zip(range(300, 1200, 300), np.linspace(0.05, 0.15, 4)):
            plt.scatter(Date_epi[obs_indices_epi], obs_data_epi[obs_indices_epi], marker='o', s=size, color=c_obs_epi, alpha=alpha, edgecolors='none')
        plt.scatter(Date_epi[obs_indices_epi], obs_data_epi[obs_indices_epi], marker='o', s=100, color=c_obs_epi, label='Obs DO conc. (epi)', alpha=0.8)


    if len(obs_indices_hypo) > 0:
        for size, alpha in zip(range(300, 1200, 300), np.linspace(0.05, 0.15, 4)): 
            plt.scatter(Date_hypo[obs_indices_hypo], obs_data_hypo[obs_indices_hypo], marker='o', s=size, color=c_obs_hyp, alpha=alpha, edgecolors='none')
        plt.scatter(Date_hypo[obs_indices_hypo], obs_data_hypo[obs_indices_hypo], marker='o', s=100, color=c_obs_hyp, label='Obs DO conc. (hyp)', alpha=0.8)

    # if len(obs_indices_mixed) > 0: 
    for size, alpha in zip(range(300, 1200, 300), np.linspace(0.05, 0.15, 4)):
        plt.scatter(Date_epi[obs_indices_mixed], obs_data_epi[obs_indices_mixed], marker='o', s=size, color=c_obs_mixed, alpha=alpha, edgecolors='none')
    plt.scatter(Date_epi[obs_indices_mixed], obs_data_epi[obs_indices_mixed], marker='o', s=100, color=c_obs_mixed, label='Obs DO conc. (total)', alpha=0.8)


    plt.gca().xaxis.set_major_locator(mdates.MonthLocator(bymonth=None, interval=3))
    plt.gca().xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))

    plt.setp(plt.gca().xaxis.get_majorticklabels(), rotation=0, ha="right")
    plt.ylabel("DO concentration ($g \, m^{-3}$)", fontsize=FONT_SIZE1)
    plt.tick_params(axis='both', which='major', labelsize=FONT_SIZE2)
    plt.legend(fontsize=FONT_SIZE_LEG, loc='upper right', frameon=True, edgecolor='black')
    plt.legend(handler_map={line_mixed_pred: CustomHandler(), line_mixed_sim: CustomHandler()}, fontsize=FONT_SIZE_LEG, loc='upper right', frameon=True, edgecolor='black')

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def draw_one_lake(seed, load_path, save_path, lake_id, pic_index,extend = False):
    pic_model_name = save_path
    # COLUMNS_USE = USE_FEATURES_COLUMNS_LAYER
    COLUMNS_USE = USE_FEATURES_COLUMNS_LAYER
    IS_PGRNN = False
    IS_FINE_TUNE = False
    USE_EXTEND = False

    ids = [lake_id]
    n_features = len(COLUMNS_USE) + 4
    use_gpu = False
    seq_length = 364
    win_shift = 364
    begin_loss_ind = 0
    if extend:
        data_dir =  f'../../data/processed_extend/seed={seed}/'
    else:
        data_dir =  f'../../data/processed_extend/seed={seed}/'

    layer_extended = 4 # Increasing the dimension representing the layer from 1 to 5.
    FLUX_START = n_features + layer_extended 
    (_, _, _, _, tst_data, tst_dates)  = buildManyLakeDataByIds(ids, data_dir, seq_length, n_features, win_shift, use_obs = True, evaluate = True)



    print(tst_data.shape)
    print(tst_dates.shape)
    batch_size = tst_data.shape[0]
    # n_hidden = torch.load(load_path, map_location=torch.device('cpu'))['state_dict']['out.weight'].shape[1]
    n_hidden = torch.load(load_path)['state_dict']['out.weight'].shape[1]
    print("n_hidden:", n_hidden)
    lstm_net = MyLSTM(n_features, n_hidden, batch_size, use_gpu)
    # pretrain_dict = torch.load(load_path, map_location=torch.device('cpu'))['state_dict']
    pretrain_dict = torch.load(load_path)['state_dict']
    model_dict = lstm_net.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    lstm_net.load_state_dict(pretrain_dict)

    #things needed to predict test data
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    testloader = torch.utils.data.DataLoader(tst_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    lstm_net.eval()
    with torch.no_grad():
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

            #run model
            h_state = None
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            pred, h_state = lstm_net(inputs, h_state)
            pred = pred.view(pred.size()[0],-1)
            pred = pred[:, begin_loss_ind:]
            pb_pred = sim_targets[:, begin_loss_ind:]
            Volume = Flux_data[:,:,0]
            Volume[1::2,:] = Flux_data[1::2,:,1]

            loss_indices = np.array(np.isfinite(obs_targets.cpu()), dtype='bool_')
            loss_sim_indices = np.array(np.isfinite(sim_targets.cpu()), dtype='bool_')

    
    print("------------------PICTURES--------------------")
    save_path_man = save_path
    if not os.path.exists(save_path_man): 
        os.makedirs(save_path_man)

    index = 0
    for i in range(0,tst_data.shape[0],2):
        epi_index = i
        hypo_index = epi_index + 1
        # pred_1_epi = pred1[epi_index]  
        pred_2_epi = pred[epi_index]  
        obs_data_epi = obs_targets[epi_index]  
        sim_data_epi = sim_targets[epi_index]
        Date_epi = tst_dates[epi_index]

        # pred_1_hypo= pred1[hypo_index]  
        pred_2_hypo = pred[hypo_index]  
        obs_data_hypo = obs_targets[hypo_index]  
        sim_data_hypo = sim_targets[hypo_index]
        Date_hypo = tst_dates[hypo_index]

        input_features = inputs[epi_index,:,:]
        mixed = input_features[:,-6]
        pred_2_hypo[np.where(mixed == 1)] = pred_2_epi[np.where(mixed == 1)]

        sim_data_hypo[np.where(mixed == 1)] = sim_data_epi[np.where(mixed == 1)]

        obs_indices_epi = np.where(loss_indices[epi_index])[0]
        sim_indices_epi = np.where(loss_sim_indices[epi_index])[0]

        obs_indices_hypo = np.where(loss_indices[hypo_index])[0]
        sim_indices_hypo = np.where(loss_sim_indices[hypo_index])[0]

        print("pred_2_hypo shape:", pred_2_hypo.shape)
        save_path = os.path.join(save_path_man, f'{pic_index}_{lake_id}_{index}.png')
        index += 1
        draw_pic(save_path, pic_model_name, Date_epi, Date_hypo, pred_2_epi, pred_2_hypo, obs_indices_epi, obs_data_epi, sim_indices_epi, sim_data_epi, obs_indices_hypo, obs_data_hypo, sim_indices_hypo, sim_data_hypo)



def draw_one_lake_beauty(seed, load_path, save_path_main, lake_id, model_name, pic_index,extend = False):
    pic_model_name = model_name
    # COLUMNS_USE = USE_FEATURES_COLUMNS_LAYER
    COLUMNS_USE = USE_FEATURES_COLUMNS_LAYER
    IS_PGRNN = False
    IS_FINE_TUNE = False
    USE_EXTEND = False

    ids = [lake_id]
    n_features = len(COLUMNS_USE) + 4
    use_gpu = False
    seq_length = 410
    win_shift = 364
    begin_loss_ind = 0
    if extend:
        data_dir =  f'../../data/processed_extend/seed={seed}/'
    else:
        data_dir =  f'../../data/processed_extend/seed={seed}/'

    layer_extended = 4 # Increasing the dimension representing the layer from 1 to 5.
    FLUX_START = n_features + layer_extended 
    (_, _, _, _, tst_data, tst_dates)  = buildManyLakeDataByIds(ids, data_dir, seq_length, n_features, win_shift, use_obs = True, evaluate = True)


    print(tst_data.shape)
    print(tst_dates.shape)
    batch_size = tst_data.shape[0]
    tst_data = tst_data[:,25:,:]
    tst_dates = tst_dates[:,25:]
    # 加载模型时，确保所有的张量都映射到CPU
    # n_hidden = torch.load(load_path, map_location=torch.device('cpu'))['state_dict']['out.weight'].shape[1]


    # load no transformer model:
    if model_name == "Transformer":
        num_heads = 5
        num_layers = 2
        hidden_size = 32
        model = TransformerModel(input_size=n_features, num_heads=num_heads, num_layers=num_layers, hidden_size=hidden_size)
        pretrain_dict = torch.load(load_path)['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        model.load_state_dict(pretrain_dict)
    else:
        n_hidden = torch.load(load_path)['state_dict']['out.weight'].shape[1]
        print("n_hidden:", n_hidden)
        model = MyLSTM(n_features, n_hidden, batch_size, use_gpu)
        # pretrain_dict = torch.load(load_path, map_location=torch.device('cpu'))['state_dict']
        pretrain_dict = torch.load(load_path)['state_dict']
        model_dict = model.state_dict()
        pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        model.load_state_dict(pretrain_dict)


    #things needed to predict test data
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    testloader = torch.utils.data.DataLoader(tst_data, batch_size=batch_size, shuffle=False, pin_memory=True)

    model.eval()
    with torch.no_grad():
        ct = 0
        for m, data in enumerate(testloader, 0):
            #now for mendota data
            #this loop is dated, there is now only one item in testloader
            #parse data into inputs and targets

            inputs = data[:,:,:n_features].float()
            print("inputs shape:", inputs.shape)
            print("tst_data shape:", tst_data.shape)
            sim_targets = data[:,:,-2].float()
            obs_targets = data[:,:,-1].float()
            Flux_data = data[:,:,FLUX_START:-2].float()
            obs_targets = obs_targets[:, begin_loss_ind:]

            #run model
            if model_name == "Transformer":
                pred = model(inputs)
            else:
                h_state = None
                model.hidden = model.init_hidden(batch_size=inputs.size()[0])
                pred, h_state = model(inputs, h_state)


            pred = pred.view(pred.size()[0],-1)
            pred = pred[:, begin_loss_ind:]
            pb_pred = sim_targets[:, begin_loss_ind:]
            Volume = Flux_data[:,:,0]
            Volume[1::2,:] = Flux_data[1::2,:,1]

            loss_indices = np.array(np.isfinite(obs_targets.cpu()), dtype='bool_')
            loss_sim_indices = np.array(np.isfinite(sim_targets.cpu()), dtype='bool_')

    print("------------------SAVE DATA--------------------")
    # save: inputs, pred, obs_targets, sim_targets, tst_dates, inputs
    save_data_path = f'../draw_pics/time-series/{lake_id}/data/'
    if not os.path.exists(save_data_path): 
        os.makedirs(save_data_path)
    np.save(save_data_path + 'inputs.npy', inputs.numpy())
    np.save(save_data_path + f'pred_{model_name}.npy', pred.numpy())
    np.save(save_data_path + 'obs_targets.npy', obs_targets.numpy())
    np.save(save_data_path + 'sim_targets.npy', sim_targets.numpy())
    np.save(save_data_path + 'tst_dates.npy', tst_dates)
    # np.savez(f'{save_data_path}' + f'{lake_id}.npz', inputs=inputs.numpy(), pred=pred.numpy(), obs_targets=obs_targets.numpy(), sim_targets=sim_targets.numpy(), tst_dates = tst_dates)

    
    print("------------------PICTURES--------------------")
    index = 0
    for i in range(0,tst_data.shape[0],2):
        epi_index = i
        hypo_index = epi_index + 1
        # pred_1_epi = pred1[epi_index]  
        pred_2_epi = pred[epi_index].clone() 
        pred_2_mixed = pred[epi_index].clone()
        obs_data_epi = obs_targets[epi_index].clone()
        sim_data_epi = sim_targets[epi_index].clone()
        sim_data_mixed = sim_targets[epi_index].clone()
        Date_epi = tst_dates[epi_index].copy()

        # pred_1_hypo= pred1[hypo_index]  
        pred_2_hypo = pred[hypo_index].clone()   
        obs_data_hypo = obs_targets[hypo_index].clone()  
        sim_data_hypo = sim_targets[hypo_index].clone()
        Date_hypo = tst_dates[hypo_index].copy()

        input_features = inputs[epi_index,:,:].clone()
        mixed = input_features[:,-6]
        pred_2_hypo[np.where(mixed == 1)] = pred_2_epi[np.where(mixed == 1)]
        sim_data_hypo[np.where(mixed == 1)] = sim_data_epi[np.where(mixed == 1)]

        obs_indices_epi = np.where(loss_indices[epi_index])[0]
        sim_indices_epi = np.where(loss_sim_indices[epi_index])[0]
        obs_indices_hypo = np.where(loss_indices[hypo_index])[0]
        sim_indices_hypo = np.where(loss_sim_indices[hypo_index])[0]



        pred_2_mixed[np.where(mixed == 0)] = np.nan
        obs_indices_mixed = obs_indices_epi[mixed[obs_indices_epi] == 1]
        obs_indices_epi = obs_indices_epi[mixed[obs_indices_epi] == 0]


        sim_indices_mixed = sim_indices_epi[mixed[sim_indices_epi] == 1]
        pred_indices_mixed = sim_indices_epi[mixed[sim_indices_epi] == 1]
        sim_data_mixed[np.where(mixed == 0)] = np.nan
        save_path = os.path.join(save_path_main, f'{pic_index}_{lake_id}_{index}_{model_name}.pdf')
        index += 1

        draw_pic_beauty(save_path, pic_model_name, Date_epi, Date_hypo, pred_2_epi, pred_2_hypo, pred_2_mixed, obs_indices_epi, obs_data_epi, sim_indices_epi, sim_data_epi, obs_indices_hypo, obs_data_hypo, sim_indices_hypo, sim_data_hypo, obs_indices_mixed, sim_data_mixed, sim_indices_mixed, pred_indices_mixed)


def draw_one_lake_beauty_new(lake_id, model_name, save_path_main, use_index):
    load_data_path = f'../draw_pics/time-series/{lake_id}/data/'

    inputs = np.load(load_data_path + 'inputs.npy')
    pred = np.load(load_data_path + f'pred_{model_name}.npy')
    obs_targets = np.load(load_data_path + 'obs_targets.npy',)
    sim_targets = np.load(load_data_path + 'sim_targets.npy')
    tst_dates = np.load(load_data_path + 'tst_dates.npy')


    loss_indices = np.array(np.isfinite(obs_targets), dtype='bool_')
    loss_sim_indices = np.array(np.isfinite(sim_targets), dtype='bool_')

    index = 0
    for i in range(0,tst_dates.shape[0],2):
        epi_index = i
        hypo_index = epi_index + 1
        # pred_1_epi = pred1[epi_index]  
        pred_2_epi = pred[epi_index].copy() 
        pred_2_mixed = pred[epi_index].copy()
        obs_data_epi = obs_targets[epi_index].copy()
        sim_data_epi = sim_targets[epi_index].copy()
        sim_data_mixed = sim_targets[epi_index].copy()
        Date_epi = tst_dates[epi_index].copy()

        # pred_1_hypo= pred1[hypo_index]  
        pred_2_hypo = pred[hypo_index].copy()   
        obs_data_hypo = obs_targets[hypo_index].copy()  
        sim_data_hypo = sim_targets[hypo_index].copy()
        Date_hypo = tst_dates[hypo_index].copy()

        input_features = inputs[epi_index,:,:].copy()
        mixed = input_features[:,-6]
        pred_2_hypo[np.where(mixed == 1)] = pred_2_epi[np.where(mixed == 1)]
        sim_data_hypo[np.where(mixed == 1)] = sim_data_epi[np.where(mixed == 1)]

        obs_indices_epi = np.where(loss_indices[epi_index])[0]
        sim_indices_epi = np.where(loss_sim_indices[epi_index])[0]
        obs_indices_hypo = np.where(loss_indices[hypo_index])[0]
        sim_indices_hypo = np.where(loss_sim_indices[hypo_index])[0]

        pred_2_mixed[np.where(mixed == 0)] = np.nan
        obs_indices_mixed = obs_indices_epi[mixed[obs_indices_epi] == 1]
        obs_indices_epi = obs_indices_epi[mixed[obs_indices_epi] == 0]


        sim_indices_mixed = sim_indices_epi[mixed[sim_indices_epi] == 1]
        pred_indices_mixed = sim_indices_epi[mixed[sim_indices_epi] == 1]
        sim_data_mixed[np.where(mixed == 0)] = np.nan
        save_path = os.path.join(save_path_main, f'{lake_id}_{index}_{model_name}.pdf')
        index += 1
        if i != use_index*2:
            continue
        draw_pic_beauty(save_path, model_name, Date_epi, Date_hypo, pred_2_epi, pred_2_hypo, pred_2_mixed, obs_indices_epi, obs_data_epi, sim_indices_epi, sim_data_epi, obs_indices_hypo, obs_data_hypo, sim_indices_hypo, sim_data_hypo, obs_indices_mixed, sim_data_mixed, sim_indices_mixed, pred_indices_mixed)
