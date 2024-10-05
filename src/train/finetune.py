from data import pytorch_data_operations
from utils.train_utils import MyLSTM, MyEALSTM, TransformerModel, set_seed, random_seed, FineTuneAprilCfg
from data.lakePreprocess import USE_FEATURES_COLUMNS, USE_FEATURES_COLUMNS_LAYER, USE_FEATURES_COLUMNS_NOFLUX, FLUX_COLUMNS
from data.pytorch_data_operations import buildOneLakeData, buildManyLakeDataByIds, calculate_total_DOC_conservation_loss, calculate_stratified_DOC_conservation_loss
from models.pytorch_model_operations import saveModel
import argparse
import json
import pickle
import random
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path, PosixPath
from typing import Dict, List, Tuple
import math
import matplotlib.pyplot as plt
import pickle as pkl
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import Dataset, DataLoader
from torch.nn.init import xavier_normal_
from torch.utils.data import DataLoader
from tqdm import tqdm
import csv
import os

# sys.path.append('../../data')
# sys.path.append('../data')
# sys.path.append('../../models')
# sys.path.append('../models')
# sys.path.append('../../utils')
# sys.path.append('../utils')

# import pytorch_data_operations
# import pytorch_model_operations

FLUX_START = -1-len(FLUX_COLUMNS)
# class MyLSTM(nn.Module):
#     def __init__(self, input_size, hidden_size, batch_size, use_gpu = False):
#         super(MyLSTM, self).__init__()
#         self.input_size = input_size
#         self.hidden_size = hidden_size
#         self.batch_size = batch_size
#         self.use_gpu = use_gpu
#         self.lstm = nn.LSTM(input_size = input_size, hidden_size=hidden_size, batch_first=True)
#         self.out = nn.Linear(hidden_size, 1)
#         self.hidden = self.init_hidden()

#     def init_hidden(self, batch_size=0):
#         # initialize both hidden layers
#         if batch_size == 0:
#             batch_size = self.batch_size
#         ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
#                 xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
#         if self.use_gpu:
#             item0 = ret[0].cuda(non_blocking=True)
#             item1 = ret[1].cuda(non_blocking=True)
#             ret = (item0,item1)
#         return ret

#     def forward(self, x, hidden):
#         self.lstm.flatten_parameters()
#         x = x.float()
#         x, hidden = self.lstm(x, self.hidden)
#         self.hidden = hidden
#         x = self.out(x)
#         x = F.relu(x)
#         return x, hidden


# Dataset classes
class OxygenTrainDataset(Dataset):
    # training dataset class, allows Dataloader to load both input/target
    def __init__(self, trn_data):
        self.len = trn_data.shape[0]
        self.x_data = trn_data[:, :, :FLUX_START].float()
        self.flux_data = trn_data[:, :, FLUX_START:-1]
        self.y_target = trn_data[:, :, -1].float()

    def __getitem__(self, index):
        return self.x_data[index], self.y_target[index], self.flux_data[index]

    def __len__(self):
        return self.len


def calculate_l1_loss(model):
    def l1_loss(x):
        return torch.abs(x).sum()

    to_regularize = []
    for name, p in model.named_parameters():
        if 'bias' in name:
            continue
        else:
            # take absolute value of weights and sum
            to_regularize.append(p.view(-1))
    # l1_loss_val = torch.tensor(1, requires_grad=True, dtype=torch.float32)
    l1_loss_val = l1_loss(torch.cat(to_regularize))
    return l1_loss_val


def train_finetune_Lstm(lake_id, group_id, seed, model_index):
    ###########
    # Globals #
    ###########
    ########################## FIXED ##########################
    IS_BUCKET = False
    seq_length = 364
    n_hidden = 50
    win_shift = 364

    begin_loss_ind = 0  # index in sequence where we begin to calculate error or predict
    grad_clip = 1.0  # how much to clip the gradient 2-norm in training
    yhat_batch_size = 2
    # hyper parameters

    ########################## UNFIXED ##########################
    use_gpu = True
    verbose = True

    IS_FINE_TUNE = True
    IS_PGRNN = False
    USE_EXTEND = False

    train_epochs = 13
    manualSeed = [random.randint(1, 99999999) for i in range(train_epochs)]
    COLUNMNS_USE = USE_FEATURES_COLUMNS_LAYER
    n_features = len(COLUNMNS_USE) + 4
    n_flux = len(FLUX_COLUMNS)
    FLUX_START = -1-len(FLUX_COLUMNS)
    doc_threshold = 0.5

    if IS_FINE_TUNE:
        use_obs = True
        model_status = 'fine_tune'
        mse_lambda = 1
        learning_rate = .005
        batch_size_ = 8

    lambda_total = 1
    lambda_stratified = 5

    if IS_FINE_TUNE:
        lambda_r1 = 0.00001  # magnitude hyperparameter of l1 loss

    cluster_id = group_id

    save_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/lstm/group_{cluster_id}/fine_tune/individual_train_on_obs/"
    if not os.path.exists(save_path_main):
        os.makedirs(save_path_main)

    model_name = f"{lake_id}_lstm_fine_tune_train_on_obs"
    save_path = save_path_main + model_name

    load_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/pre_train/group_{cluster_id}/pre_train_LSTM/lambda=0/"
    load_model_name = f"group_{cluster_id}_pre_train_model_pre_train_LSTM_41_{model_index}_lambda-0_train_on_obs"
    load_path = load_path_main + load_model_name
    # print("load path:", load_path)
    # print("save path:",save_path)
    # print("colunmns_use size:", len(COLUNMNS_USE))

    if not USE_EXTEND:
        data_dir = f"../../data/processed/"
    else:
        data_dir = f"../../data/processed_extend/"

    if not IS_FINE_TUNE:
        ids = pd.read_csv(
            f'../../data/utils/groups/vol_area/cluster_{cluster_id}.csv')
        ids = ids['nhdhr_id'].to_list()
    else:
        ids = [lake_id]
        print("ids:", ids)

    (train_data, trn_dates, val_data, val_dates, tst_data, tst_dates) = buildManyLakeDataByIds(
        ids, data_dir, seq_length, n_features, win_shift, use_obs, evaluate=False)

    print("Before:")
    print(train_data.shape)
    print(val_data.shape)
    print(tst_data.shape)

    if not IS_FINE_TUNE:
        trn_data = torch.cat((train_data, val_data), dim=0)
        trn_data = torch.cat((trn_data, tst_data), dim=0)
        print("After concatenate Test and Train:")
        print(trn_data.shape)
    else:
        trn_data = train_data

    unsup_data = torch.cat((train_data, val_data), dim=0)
    unsup_data = torch.cat((unsup_data, tst_data), dim=0)
    print(unsup_data.shape)
    batch_size = batch_size_

    n_batches = math.floor(trn_data.size()[0] / batch_size)
    n_val_batches = math.floor(val_data.size()[0] / batch_size)
    n_batches_unsup = math.floor(unsup_data.size()[0] / batch_size)
    train_data = OxygenTrainDataset(trn_data)
    val_data = OxygenTrainDataset(val_data)
    unsup_data = OxygenTrainDataset(unsup_data)

    n_depths = 2
    yhat_batch_size = n_depths

    batch_sampler = pytorch_data_operations.ContiguousBatchSampler(
        batch_size, n_batches)
    print("Tranining batch size:", batch_size)

    lstm_net = MyLSTM(n_features, n_hidden, batch_size, use_gpu)
    # tell model to use GPU if needed

    if IS_FINE_TUNE:
        pretrain_dict = torch.load(load_path)['state_dict']
        model_dict = lstm_net.state_dict()
        pretrain_dict = {k: v for k,
                         v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        lstm_net.load_state_dict(pretrain_dict)

    if use_gpu:
        lstm_net = lstm_net.cuda()

    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_net.parameters(), lr=learning_rate)

    # convergence variables
    converged = False

    training_errors = []
    validation_errors = []
    for epoch in range(train_epochs):
        if verbose:
            print("train epoch: ", epoch)
        if use_gpu:
            set_seed(manualSeed[epoch])
        running_loss = 0.0

        # val_batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_val_batches)
        # valloader = DataLoader(val_data, batch_sampler=val_batch_sampler, pin_memory=True)

        batch_sampler = pytorch_data_operations.ContiguousBatchSampler(
            batch_size, n_batches)
        batch_sampler_unsup = pytorch_data_operations.RandomContiguousBatchSampler(
            batch_size, 2, n_batches_unsup)  # yhat_batch_size = layer number

        unsupdataloader = DataLoader(
            unsup_data, batch_sampler=batch_sampler_unsup, pin_memory=True)
        trainloader = DataLoader(
            train_data, batch_sampler=batch_sampler, pin_memory=True)
        multi_loader = pytorch_data_operations.MultiLoader(
            [trainloader, unsupdataloader])

        # zero the parameter gradients
        optimizer.zero_grad()
        lstm_net.train(True)
        avg_loss = 0
        avg_total_DOC_conservation_loss = 0
        avg_stratified_DOC_conservation_loss = 0
        batches_done = 0
        for i, batches in enumerate(multi_loader):

            # load data
            inputs = None
            targets = None

            unsup_inputs = None
            flux_data = None
            for j, b in enumerate(batches):
                if j == 0:
                    inputs, targets, Flux_data = b

                if j == 1:
                    unsup_inputs, _, flux_data = b

            Volume = Flux_data[:, :, 0]
            Volume[1::2, :] = Flux_data[1::2, :, 1]

            if (use_gpu):
                inputs = inputs.cuda()
                targets = targets.cuda()
                Volume = Volume.cuda()

            # forward  prop
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            h_state = None
            outputs, h_state = lstm_net(inputs, h_state)
            outputs = outputs.view(outputs.size()[0], -1)
            assert Volume.shape == outputs.shape, "Shape not match"

            # unsupervised output
            h_state = None
            lstm_net.hidden = lstm_net.init_hidden(
                batch_size=unsup_inputs.size()[0])
            total_DOC_conservation_loss = torch.tensor(0).float()
            stratified_DOC_conservation_loss = torch.tensor(0).float()
            if use_gpu:
                unsup_inputs = unsup_inputs.cuda()
                flux_data = flux_data.cuda()
                total_DOC_conservation_loss = total_DOC_conservation_loss.cuda()
                stratified_DOC_conservation_loss = stratified_DOC_conservation_loss.cuda()

            # get unsupervised outputs
            unsup_outputs, h_state = lstm_net(unsup_inputs, h_state)

            if use_gpu:
                unsup_outputs = unsup_outputs.cuda()

            # calculate unsupervised loss
            if IS_FINE_TUNE and IS_PGRNN:
                for index in range(0, unsup_inputs.shape[0], 2):
                    # unsup_inputs.shape[0] == 2 !!, so there will only be one round
                    total_DOC_conservation_loss += (calculate_total_DOC_conservation_loss(
                        flux_data[index:index+2, :, :], unsup_outputs[index:index+2, :, :], doc_threshold, 1, use_gpu) / (unsup_inputs.shape[0]/2))
                    stratified_DOC_conservation_loss += (calculate_stratified_DOC_conservation_loss(
                        flux_data[index:index+2, :, :], unsup_outputs[index:index+2, :, :], doc_threshold, 1, use_gpu) / (unsup_inputs.shape[0]/2))
            else:
                total_DOC_conservation_loss = 0
                stratified_DOC_conservation_loss = 0

            # total_DOC_conservation_loss = calculate_total_DOC_conservation_loss(flux_data, unsup_outputs, doc_threshold, 1, use_gpu)
            # stratified_DOC_conservation_loss = calculate_stratified_DOC_conservation_loss(flux_data, unsup_outputs, doc_threshold, 1, use_gpu)

            # calculate supervised loss
            loss_outputs = outputs[:, begin_loss_ind:]
            loss_targets = targets[:, begin_loss_ind:]
            # open when training LSTM model with obs
            # loss_targets = obs_target[:,begin_loss_ind:]

            # get indices to calculate loss
            loss_indices = np.array(np.isfinite(
                loss_targets.cpu()), dtype='bool_')

            if use_gpu:
                loss_outputs = loss_outputs.cuda()
                loss_targets = loss_targets.cuda()

            supervised_loss = mse_criterion(
                loss_outputs[loss_indices], loss_targets[loss_indices])
            # R1 loss
            if lambda_r1 > 0:
                reg1_loss = calculate_l1_loss(lstm_net)
            else:
                reg1_loss = 0

            if torch.isnan(loss_targets).all():
                continue
                # print("loss_targets中全部都是NaN")

            # total loss
            loss = mse_lambda*supervised_loss + lambda_r1*reg1_loss + lambda_total * \
                total_DOC_conservation_loss + lambda_stratified * stratified_DOC_conservation_loss
            # loss = mse_lambda*supervised_loss + lambda_r1*reg1_loss + 0 * total_DOC_conservation_loss + 0 * stratified_DOC_conservation_loss
            # backward
            loss.backward(retain_graph=False)
            if grad_clip > 0:
                clip_grad_norm_(lstm_net.parameters(), grad_clip, norm_type=2)

            # optimize
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
            avg_loss += loss
            avg_total_DOC_conservation_loss += total_DOC_conservation_loss
            avg_stratified_DOC_conservation_loss += stratified_DOC_conservation_loss
            batches_done += 1
            # print(f"epoch:{epoch},batch:{i}, un_loss = :{unsup_loss}")

        avg_loss = avg_loss / batches_done
        avg_total_DOC_conservation_loss = lambda_total * \
            avg_total_DOC_conservation_loss / batches_done
        avg_stratified_DOC_conservation_loss = lambda_stratified * \
            avg_stratified_DOC_conservation_loss / batches_done
        training_errors.append(avg_loss.cpu().item())

        # Val dataset
        # val_loss = 0
        # with torch.no_grad():
        #     lstm_net.eval()
        #     batches_done = 0
        #     for inputs, targets, obs_targets, _ in valloader:

        #         if use_gpu:
        #             inputs, targets, obs_targets = inputs.cuda(), targets.cuda(), obs_targets.cuda()

        #         if IS_FINE_TUNE:
        #             loss_targets = obs_targets
        #         else:
        #             loss_targets = targets

        #         # loss_targets = obs_targets
        #         loss_indices = np.array(np.isfinite(loss_targets.cpu()), dtype='bool_')
        #         h_state = None
        #         lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
        #         outputs, h_state = lstm_net(inputs, h_state)
        #         outputs = outputs.view(outputs.size()[0], -1)
        #         val_loss += mse_criterion(outputs[loss_indices], loss_targets[loss_indices])
        #         batches_done+=1

        #     val_loss = val_loss / batches_done
        #     validation_errors.append(val_loss.cpu().item())

        if verbose:
            print("Supervised: ")
            print("Training loss=", avg_loss)

            print("Unsupervised: ")
            print("Total_DOC_conservation loss =",
                  avg_total_DOC_conservation_loss)
            print("Stratified_DOC_conservation loss =",
                  avg_stratified_DOC_conservation_loss)
            # print("Validation:")
            # print("Validation loss =",val_loss)
            print("=======================")

        if avg_loss < 1:
            if verbose:
                print("training converged")
            converged = True

        if converged:
            saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
            print("training finished in ", epoch)
            break
        # break
    print("TRAINING COMPLETE")
    saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

def train_finetune_Pril(lake_id, group_id, seed, model_index):
    ###########
    # Globals #
    ###########
    ########################## FIXED ##########################
    IS_BUCKET = False
    seq_length = 364
    n_hidden = 50
    win_shift = 364

    begin_loss_ind = 0  # index in sequence where we begin to calculate error or predict
    grad_clip = 1.0  # how much to clip the gradient 2-norm in training
    yhat_batch_size = 2
    # hyper parameters

    ########################## UNFIXED ##########################
    use_gpu = True
    verbose = True

    IS_FINE_TUNE = True
    IS_PGRNN = True
    USE_EXTEND = False

    if USE_EXTEND:
        use_extend = '_use_extend'
    else:
        use_extend = ''

    train_epochs = 10
    manualSeed = [random.randint(1, 99999999) for i in range(train_epochs)]
    COLUNMNS_USE = USE_FEATURES_COLUMNS_LAYER
    n_features = len(COLUNMNS_USE) + 4
    FLUX_START = -1-len(FLUX_COLUMNS)
    doc_threshold = 0

    if IS_FINE_TUNE:
        use_obs = True
        model_status = 'fine_tune'
        mse_lambda = 1
        learning_rate = .025
        batch_size_ = 8


    lambda_total = 1
    lambda_stratified = 5
    lambda_r1 = 0  # magnitude hyperparameter of l1 loss
    cluster_id = group_id
    save_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/pril/group_{cluster_id}/fine_tune/individual_train_on_obs/"
    if not os.path.exists(save_path_main):
        os.makedirs(save_path_main)

    model_name = f"{lake_id}_pril_fine_tune_train_on_obs"
    save_path = save_path_main + model_name

    load_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/pre_train/group_{cluster_id}/pre_train_PRIL/lambda=0/"
    load_model_name = f"group_{cluster_id}_pre_train_model_pre_train_PRIL_41_{model_index}_lambda-0_train_on_obs"
    load_path = load_path_main + load_model_name
    # print("load path:", load_path)
    # print("save path:",save_path)
    # print("colunmns_use size:", len(COLUNMNS_USE))

    data_dir = f"../../data/processed/"

    if not IS_FINE_TUNE:
        ids = pd.read_csv(
            f'../../data/utils/groups/vol_area/cluster_{cluster_id}.csv')
        ids = ids['nhdhr_id'].to_list()
    else:
        ids = [lake_id]
        print("ids:", ids)

    (train_data, trn_dates, val_data, val_dates, tst_data, tst_dates) = buildManyLakeDataByIds(
        ids, data_dir, seq_length, n_features, win_shift, use_obs, evaluate=False)

    print("Before:")
    print(train_data.shape)
    print(val_data.shape)
    print(tst_data.shape)

    if not IS_FINE_TUNE:
        trn_data = torch.cat((train_data, val_data), dim=0)
        trn_data = torch.cat((trn_data, tst_data), dim=0)
        print("After concatenate Test and Train:")
        print(trn_data.shape)
    else:
        trn_data = train_data

    # unsup_data = torch.cat((train_data, val_data), dim=0)
    # unsup_data = torch.cat((unsup_data, tst_data), dim=0)

    unsup_data = tst_data
    unsup_data_size = unsup_data.shape[0]
    print(unsup_data.shape)
    batch_size = batch_size_

    n_batches = math.floor(trn_data.size()[0] / batch_size)
    n_val_batches = math.floor(val_data.size()[0] / batch_size)
    n_batches_unsup = math.floor(unsup_data.size()[0] / batch_size)
    train_data = OxygenTrainDataset(trn_data)
    val_data = OxygenTrainDataset(val_data)
    unsup_data = OxygenTrainDataset(unsup_data)

    n_depths = 2
    yhat_batch_size = n_depths

    batch_sampler = pytorch_data_operations.ContiguousBatchSampler(
        batch_size, n_batches)
    print("Tranining batch size:", batch_size)

    lstm_net = MyLSTM(n_features, n_hidden, batch_size, use_gpu)
    # tell model to use GPU if needed

    if IS_FINE_TUNE:
        pretrain_dict = torch.load(load_path)['state_dict']
        model_dict = lstm_net.state_dict()
        pretrain_dict = {k: v for k,
                         v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        lstm_net.load_state_dict(pretrain_dict)

    if use_gpu:
        lstm_net = lstm_net.cuda()

    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_net.parameters(), lr=learning_rate)

    # convergence variables
    converged = False

    training_errors = []
    validation_errors = []
    for epoch in range(train_epochs):
        if verbose:
            print("train epoch: ", epoch)
        if use_gpu:
            set_seed(manualSeed[epoch])
        running_loss = 0.0

        # val_batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_val_batches)
        # valloader = DataLoader(val_data, batch_sampler=val_batch_sampler, pin_memory=True)

        batch_sampler = pytorch_data_operations.ContiguousBatchSampler(
            batch_size, n_batches)
        batch_sampler_unsup = pytorch_data_operations.RandomContiguousBatchSampler(
            unsup_data_size, 2, n_batches)  # yhat_batch_size = layer number

        unsupdataloader = DataLoader(
            unsup_data, batch_sampler=batch_sampler_unsup, pin_memory=True)
        trainloader = DataLoader(
            train_data, batch_sampler=batch_sampler, pin_memory=True)
        multi_loader = pytorch_data_operations.MultiLoader(
            [trainloader, unsupdataloader])

        # zero the parameter gradients
        optimizer.zero_grad()
        lstm_net.train(True)
        avg_loss = 0
        avg_total_DOC_conservation_loss = 0
        avg_upper_DOC_conservation_loss = 0
        avg_lower_DOC_conservation_loss = 0

        batches_done = 0
        for i, batches in enumerate(multi_loader):

            # load data
            inputs = None
            targets = None

            unsup_inputs = None
            flux_data = None
            for j, b in enumerate(batches):
                if j == 0:
                    inputs, targets, Flux_data = b

                if j == 1:
                    unsup_inputs, _, flux_data = b

            Volume = Flux_data[:, :, 0]
            Volume[1::2, :] = Flux_data[1::2, :, 1]

            if (use_gpu):
                inputs = inputs.cuda()
                targets = targets.cuda()
                Volume = Volume.cuda()

            # forward  prop
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            h_state = None
            outputs, h_state = lstm_net(inputs, h_state)
            outputs = outputs.view(outputs.size()[0], -1)
            assert Volume.shape == outputs.shape, "Shape not match"

            # unsupervised output
            h_state = None
            lstm_net.hidden = lstm_net.init_hidden(
                batch_size=unsup_inputs.size()[0])
            total_DOC_conservation_loss = torch.tensor(0).float()
            upper_DOC_conservation_loss = torch.tensor(0).float()
            lower_DOC_conservation_loss = torch.tensor(0).float()
            if use_gpu:
                unsup_inputs = unsup_inputs.cuda()
                flux_data = flux_data.cuda()
                total_DOC_conservation_loss = total_DOC_conservation_loss.cuda()
                upper_DOC_conservation_loss = upper_DOC_conservation_loss.cuda()
                lower_DOC_conservation_loss = lower_DOC_conservation_loss.cuda()

                # stratified_DOC_conservation_loss = stratified_DOC_conservation_loss.cuda()

            # get unsupervised outputs
            unsup_outputs, h_state = lstm_net(unsup_inputs, h_state)

            if use_gpu:
                unsup_outputs = unsup_outputs.cuda()

            # calculate unsupervised loss
            if IS_FINE_TUNE and IS_PGRNN:
                for index in range(0, unsup_inputs.shape[0], 2):
                    # unsup_inputs.shape[0] == 2 !!, so there will only be one round
                    total_DOC_conservation_loss += (calculate_total_DOC_conservation_loss(
                        flux_data[index:index+2, :, :], unsup_outputs[index:index+2, :, :], doc_threshold, 1, use_gpu) / (unsup_inputs.shape[0]/2))
                    upper_loss, lower_loss = calculate_stratified_DOC_conservation_loss(
                        flux_data[index:index+2, :, :], unsup_outputs[index:index+2, :, :], doc_threshold, 1, use_gpu)
                    upper_DOC_conservation_loss += upper_loss / \
                        (unsup_inputs.shape[0]/2)
                    lower_DOC_conservation_loss += lower_loss / \
                        (unsup_inputs.shape[0]/2)

            else:
                total_DOC_conservation_loss = 0
                # stratified_DOC_conservation_loss = 0

            # total_DOC_conservation_loss = calculate_total_DOC_conservation_loss(flux_data, unsup_outputs, doc_threshold, 1, use_gpu)
            # stratified_DOC_conservation_loss = calculate_stratified_DOC_conservation_loss(flux_data, unsup_outputs, doc_threshold, 1, use_gpu)

            # calculate supervised loss
            loss_outputs = outputs[:, begin_loss_ind:]
            loss_targets = targets[:, begin_loss_ind:]
            # open when training LSTM model with obs
            # loss_targets = obs_target[:,begin_loss_ind:]

            # get indices to calculate loss
            loss_indices = np.array(np.isfinite(
                loss_targets.cpu()), dtype='bool_')

            if use_gpu:
                loss_outputs = loss_outputs.cuda()
                loss_targets = loss_targets.cuda()

            supervised_loss = mse_criterion(
                loss_outputs[loss_indices], loss_targets[loss_indices])
            # R1 loss
            if lambda_r1 > 0:
                reg1_loss = calculate_l1_loss(lstm_net)
            else:
                reg1_loss = 0

            if torch.isnan(loss_targets).all():
                continue
                # print("loss_targets中全部都是NaN")

            # total loss
            loss = mse_lambda*supervised_loss + lambda_r1*reg1_loss + lambda_total * total_DOC_conservation_loss + \
                lambda_stratified * \
                (upper_DOC_conservation_loss + lower_DOC_conservation_loss)
            # loss = mse_lambda*supervised_loss + lambda_r1*reg1_loss + 0 * total_DOC_conservation_loss + 0 * stratified_DOC_conservation_loss
            # backward
            loss.backward(retain_graph=False)
            if grad_clip > 0:
                clip_grad_norm_(lstm_net.parameters(), grad_clip, norm_type=2)

            # optimize
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
            avg_loss += loss
            avg_total_DOC_conservation_loss += total_DOC_conservation_loss
            avg_upper_DOC_conservation_loss += upper_DOC_conservation_loss
            avg_lower_DOC_conservation_loss += lower_DOC_conservation_loss
            batches_done += 1
            # print(f"epoch:{epoch},batch:{i}, un_loss = :{unsup_loss}")

        avg_loss = avg_loss / batches_done
        avg_total_DOC_conservation_loss = lambda_total * \
            avg_total_DOC_conservation_loss / batches_done
        avg_upper_DOC_conservation_loss = lambda_stratified * \
            avg_upper_DOC_conservation_loss / batches_done
        avg_lower_DOC_conservation_loss = lambda_stratified * \
            avg_lower_DOC_conservation_loss / batches_done
        training_errors.append(avg_loss.cpu().item())
        if verbose:
            print("Supervised: ")
            print("Training loss=", avg_loss)

            print("Unsupervised: ")
            print("Total_DOC_conservation loss =",
                  avg_total_DOC_conservation_loss)
            print("Upper layer Stratified_DOC_conservation loss =",
                  avg_upper_DOC_conservation_loss)
            print("Lower layer Stratified_DOC_conservation loss =",
                  avg_lower_DOC_conservation_loss)
            # print("Validation:")
            # print("Validation loss =",val_loss)
            print("=======================")

        if avg_loss < 1:
            if verbose:
                print("training converged")
            converged = True

        if converged:
            saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
            print("training finished in ", epoch)
            break
        # break
    print("TRAINING COMPLETE")
    saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

def train_finetune_April(lake_id, group_id, seed, model_index):
    ###########
    # Globals #
    ###########
    ########################## FIXED ##########################
    IS_BUCKET = False
    seq_length = 364
    n_hidden = 50
    win_shift = 364

    begin_loss_ind = 0  # index in sequence where we begin to calculate error or predict
    grad_clip = 1.0  # how much to clip the gradient 2-norm in training
    yhat_batch_size = 2
    # hyper parameters

    ########################## UNFIXED ##########################
    use_gpu = True
    verbose = True

    IS_FINE_TUNE = True
    IS_PGRNN = True
    USE_EXTEND = True

    cfg = FineTuneAprilCfg(group_id)

    if IS_FINE_TUNE:
        train_epochs = cfg.train_epochs

    manualSeed = [random.randint(1, 99999999) for i in range(train_epochs)]
    COLUNMNS_USE = USE_FEATURES_COLUMNS_LAYER
    n_features = len(COLUNMNS_USE) + 4
    n_flux = len(FLUX_COLUMNS)
    FLUX_START = -1-len(FLUX_COLUMNS)
    doc_threshold = 0.0

    if IS_FINE_TUNE:
        use_obs = True
        model_status = 'fine_tune'
        mse_lambda = 1
        learning_rate = cfg.learning_rate
        batch_size_ = 8

    lambda_total = cfg.lambda_total
    lambda_stratified_epi = cfg.lambda_stratified_epi
    lambda_stratified_hypo = cfg.lambda_stratified_hypo
    if IS_FINE_TUNE:
        lambda_r1 = 0  # magnitude hyperparameter of l1 loss
    else:
        model_type = 'pre_train'
        lambda_r1 = 0.0001  # magnitude hyperparameter of l1 loss

    cluster_id = group_id

    save_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/april/group_{cluster_id}/fine_tune/individual_train_on_obs_extend/"
    if not os.path.exists(save_path_main):
        os.makedirs(save_path_main)

    model_name = f"{lake_id}_april_fine_tune_train_on_obs_12k"
    save_path = save_path_main + model_name

    load_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/pre_train/group_{cluster_id}/pre_train_APRIL/lambda=0/"
    load_model_name = f"group_{cluster_id}_pre_train_model_pre_train_APRIL_41_{model_index}_lambda-0_use_extend_train_on_obs"
    load_path = load_path_main + load_model_name

    if not USE_EXTEND:
        data_dir = f"../../data/processed/"
    else:
        data_dir = f"../../data/processed_extend/seed={seed}/"

    if not IS_FINE_TUNE:
        ids = pd.read_csv(
            f'../../data/utils/groups/vol_area/cluster_{cluster_id}.csv')
        ids = ids['nhdhr_id'].to_list()
    else:
        ids = [lake_id]
        print("ids:", ids)

    (train_data, trn_dates, val_data, val_dates, tst_data, tst_dates) = buildManyLakeDataByIds(
        ids, data_dir, seq_length, n_features, win_shift, use_obs, evaluate=False)

    print("Before:")
    print(train_data.shape)
    print(val_data.shape)
    print(tst_data.shape)

    if not IS_FINE_TUNE:
        trn_data = torch.cat((train_data, val_data), dim=0)
        trn_data = torch.cat((trn_data, tst_data), dim=0)
        print("After concatenate Test and Train:")
        print(trn_data.shape)
    else:
        trn_data = train_data

    # unsup_data = torch.cat((train_data, val_data), dim=0)
    # unsup_data = torch.cat((unsup_data, tst_data), dim=0)

    unsup_data = tst_data
    unsup_data_size = unsup_data.shape[0]
    print(unsup_data.shape)
    batch_size = batch_size_

    n_batches = math.floor(trn_data.size()[0] / batch_size)
    n_val_batches = math.floor(val_data.size()[0] / batch_size)
    n_batches_unsup = math.floor(unsup_data.size()[0] / batch_size)
    train_data = OxygenTrainDataset(trn_data)
    val_data = OxygenTrainDataset(val_data)
    unsup_data = OxygenTrainDataset(unsup_data)

    n_depths = 2
    yhat_batch_size = n_depths

    batch_sampler = pytorch_data_operations.ContiguousBatchSampler(
        batch_size, n_batches)
    print("Tranining batch size:", batch_size)

    lstm_net = MyLSTM(n_features, n_hidden, batch_size, use_gpu)
    # tell model to use GPU if needed

    if IS_FINE_TUNE:
        pretrain_dict = torch.load(load_path)['state_dict']
        model_dict = lstm_net.state_dict()
        pretrain_dict = {k: v for k,
                         v in pretrain_dict.items() if k in model_dict}
        model_dict.update(pretrain_dict)
        lstm_net.load_state_dict(pretrain_dict)

    if use_gpu:
        lstm_net = lstm_net.cuda()

    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(lstm_net.parameters(), lr=learning_rate)

    # convergence variables
    converged = False

    training_errors = []
    validation_errors = []
    for epoch in range(train_epochs):
        if verbose:
            print("train epoch: ", epoch)
        if use_gpu:
            set_seed(manualSeed[epoch])
        running_loss = 0.0

        # val_batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_val_batches)
        # valloader = DataLoader(val_data, batch_sampler=val_batch_sampler, pin_memory=True)

        batch_sampler = pytorch_data_operations.ContiguousBatchSampler(
            batch_size, n_batches)
        batch_sampler_unsup = pytorch_data_operations.RandomContiguousBatchSampler(
            unsup_data_size, 2, n_batches)  # yhat_batch_size = layer number

        unsupdataloader = DataLoader(
            unsup_data, batch_sampler=batch_sampler_unsup, pin_memory=True)
        trainloader = DataLoader(
            train_data, batch_sampler=batch_sampler, pin_memory=True)
        multi_loader = pytorch_data_operations.MultiLoader(
            [trainloader, unsupdataloader])

        # zero the parameter gradients
        optimizer.zero_grad()
        lstm_net.train(True)
        avg_loss = 0
        avg_total_DOC_conservation_loss = 0
        avg_upper_DOC_conservation_loss = 0
        avg_lower_DOC_conservation_loss = 0

        batches_done = 0
        for i, batches in enumerate(multi_loader):

            # load data
            inputs = None
            targets = None

            unsup_inputs = None
            flux_data = None
            for j, b in enumerate(batches):
                if j == 0:
                    inputs, targets, Flux_data = b

                if j == 1:
                    unsup_inputs, _, flux_data = b

            Volume = Flux_data[:, :, 0]
            Volume[1::2, :] = Flux_data[1::2, :, 1]

            if (use_gpu):
                inputs = inputs.cuda()
                targets = targets.cuda()
                Volume = Volume.cuda()

            # forward  prop
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            h_state = None
            outputs, h_state = lstm_net(inputs, h_state)
            outputs = outputs.view(outputs.size()[0], -1)
            assert Volume.shape == outputs.shape, "Shape not match"

            # unsupervised output
            h_state = None
            lstm_net.hidden = lstm_net.init_hidden(
                batch_size=unsup_inputs.size()[0])
            total_DOC_conservation_loss = torch.tensor(0).float()
            upper_DOC_conservation_loss = torch.tensor(0).float()
            lower_DOC_conservation_loss = torch.tensor(0).float()
            if use_gpu:
                unsup_inputs = unsup_inputs.cuda()
                flux_data = flux_data.cuda()
                total_DOC_conservation_loss = total_DOC_conservation_loss.cuda()
                upper_DOC_conservation_loss = upper_DOC_conservation_loss.cuda()
                lower_DOC_conservation_loss = lower_DOC_conservation_loss.cuda()

                # stratified_DOC_conservation_loss = stratified_DOC_conservation_loss.cuda()

            # get unsupervised outputs
            unsup_outputs, h_state = lstm_net(unsup_inputs, h_state)

            if use_gpu:
                unsup_outputs = unsup_outputs.cuda()

            # calculate unsupervised loss
            if IS_FINE_TUNE and IS_PGRNN:
                for index in range(0, unsup_inputs.shape[0], 2):
                    # unsup_inputs.shape[0] == 2 !!, so there will only be one round
                    total_DOC_conservation_loss += (calculate_total_DOC_conservation_loss(
                        flux_data[index:index+2, :, :], unsup_outputs[index:index+2, :, :], doc_threshold, 1, use_gpu) / (unsup_inputs.shape[0]/2))
                    upper_loss, lower_loss = calculate_stratified_DOC_conservation_loss(
                        flux_data[index:index+2, :, :], unsup_outputs[index:index+2, :, :], doc_threshold, 1, use_gpu)
                    upper_DOC_conservation_loss += upper_loss / \
                        (unsup_inputs.shape[0]/2)
                    lower_DOC_conservation_loss += lower_loss / \
                        (unsup_inputs.shape[0]/2)

            else:
                total_DOC_conservation_loss = 0
                # stratified_DOC_conservation_loss = 0

            # total_DOC_conservation_loss = calculate_total_DOC_conservation_loss(flux_data, unsup_outputs, doc_threshold, 1, use_gpu)
            # stratified_DOC_conservation_loss = calculate_stratified_DOC_conservation_loss(flux_data, unsup_outputs, doc_threshold, 1, use_gpu)

            # calculate supervised loss
            loss_outputs = outputs[:, begin_loss_ind:]
            loss_targets = targets[:, begin_loss_ind:]

            # open when training LSTM model with obs
            # loss_targets = obs_target[:,begin_loss_ind:]

            # get indices to calculate loss
            loss_indices = np.array(np.isfinite(
                loss_targets.cpu()), dtype='bool_')

            if use_gpu:
                loss_outputs = loss_outputs.cuda()
                loss_targets = loss_targets.cuda()

            supervised_loss = mse_criterion(
                loss_outputs[loss_indices], loss_targets[loss_indices])
            # R1 loss
            if lambda_r1 > 0:
                reg1_loss = calculate_l1_loss(lstm_net)
            else:
                reg1_loss = 0

            if torch.isnan(loss_targets).all():
                continue
                # print("loss_targets中全部都是NaN")

            # total loss
            loss = mse_lambda*supervised_loss + lambda_r1*reg1_loss + lambda_total * total_DOC_conservation_loss + \
                lambda_stratified_epi * upper_DOC_conservation_loss + \
                lambda_stratified_hypo * lower_DOC_conservation_loss
            # loss = mse_lambda*supervised_loss + lambda_r1*reg1_loss + 0 * total_DOC_conservation_loss + 0 * stratified_DOC_conservation_loss
            # backward
            loss.backward(retain_graph=False)
            if grad_clip > 0:
                clip_grad_norm_(lstm_net.parameters(), grad_clip, norm_type=2)

            # optimize
            optimizer.step()
            # zero the parameter gradients
            optimizer.zero_grad()
            avg_loss += loss
            avg_total_DOC_conservation_loss += total_DOC_conservation_loss
            avg_upper_DOC_conservation_loss += upper_DOC_conservation_loss
            avg_lower_DOC_conservation_loss += lower_DOC_conservation_loss
            batches_done += 1
            # print(f"epoch:{epoch},batch:{i}, un_loss = :{unsup_loss}")

        avg_loss = avg_loss / batches_done
        avg_total_DOC_conservation_loss = avg_total_DOC_conservation_loss / batches_done
        avg_upper_DOC_conservation_loss = avg_upper_DOC_conservation_loss / batches_done
        avg_lower_DOC_conservation_loss = avg_lower_DOC_conservation_loss / batches_done
        training_errors.append(avg_loss.cpu().item())

        # Val dataset
        # val_loss = 0
        # with torch.no_grad():
        #     lstm_net.eval()
        #     batches_done = 0
        #     for inputs, targets, obs_targets, _ in valloader:

        #         if use_gpu:
        #             inputs, targets, obs_targets = inputs.cuda(), targets.cuda(), obs_targets.cuda()

        #         if IS_FINE_TUNE:
        #             loss_targets = obs_targets
        #         else:
        #             loss_targets = targets

        #         # loss_targets = obs_targets
        #         loss_indices = np.array(np.isfinite(loss_targets.cpu()), dtype='bool_')
        #         h_state = None
        #         lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
        #         outputs, h_state = lstm_net(inputs, h_state)
        #         outputs = outputs.view(outputs.size()[0], -1)
        #         val_loss += mse_criterion(outputs[loss_indices], loss_targets[loss_indices])
        #         batches_done+=1

        #     val_loss = val_loss / batches_done
        #     validation_errors.append(val_loss.cpu().item())

        if verbose:
            print("Supervised: ")
            print("Training loss=", avg_loss)

            print("Unsupervised: ")
            print("Total_DOC_conservation loss =",
                  avg_total_DOC_conservation_loss)
            print("Upper layer Stratified_DOC_conservation loss =",
                  avg_upper_DOC_conservation_loss)
            print("Lower layer Stratified_DOC_conservation loss =",
                  avg_lower_DOC_conservation_loss)
            # print("Validation:")
            # print("Validation loss =",val_loss)
            print("=======================")

        if avg_loss < 1:
            if verbose:
                print("training converged")
            converged = True

        if converged:
            saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
            print("training finished in ", epoch)
            break
        # break
    print("TRAINING COMPLETE")
    saveModel(lstm_net.state_dict(), optimizer.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

def train_finetune_EaLstm(lake_id, group_id, seed, model_index):
    COLUNMNS_USE = USE_FEATURES_COLUMNS_LAYER
    seq_length = 364
    n_hidden = 50
    win_shift = 364

    begin_loss_ind = 0#index in sequence where we begin to calculate error or predict
    grad_clip = 1.0 #how much to clip the gradient 2-norm in training
    yhat_batch_size = 2
    # hyper parameters

    ########################## UNFIXED ##########################
    use_gpu = True
    verbose = True

    IF_PRETRAIN = True
    IS_PGRNN = True

    train_epochs = 10
    n_features = len(COLUNMNS_USE) + 4
    n_flux = len(FLUX_COLUMNS)
    FLUX_START = -1-len(FLUX_COLUMNS)
    doc_threshold = 0.5

    use_obs = True
    doc_lambda = 0
    mse_lambda = 1
    learning_rate = .005
    batch_size_ = 8

    model_type = 'ea_lstm'
    model_status = 'fine_tune'
    lambda_r1 = 0 #magnitude hyperparameter of l1 loss

    cluster_id = group_id
    model_index = model_index

    save_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/{model_type}/group_{cluster_id}/{model_status}/lambda={doc_lambda}/individual_train_on_obs/"
    if not os.path.exists(save_path_main): 
        os.makedirs(save_path_main)

    model_name = f"{lake_id}_ealstm_index_{model_index}_fine_tune_train_on_obs"
    save_path = save_path_main+ model_name
    print("save path:",save_path)

    ###############################
    # data preprocess
    ##################################
    #create train and test sets

    data_dir = f"../../data/processed/" # 
    ids = [lake_id]

    (train_data, trn_dates, val_data, val_dates, tst_data, tst_dates)  = buildManyLakeDataByIds(ids, data_dir, seq_length, n_features, win_shift, use_obs, evaluate=False)

    print("Training shape",train_data.shape)
    print("Validation shape",val_data.shape)
    print("Testing shape",tst_data.shape)
    print("=============")

    trn_data = train_data
    batch_size = batch_size_
    n_batches =  math.floor(trn_data.size()[0] / batch_size) 
    train_data = OxygenTrainDataset(trn_data)
    val_data = OxygenTrainDataset(val_data)
    print("n_batches:", n_batches)
    n_depths = 2
    yhat_batch_size = n_depths
    batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)
    print("Tranining batch size:", batch_size)

    input_size_dyn = 35
    input_size_stat = 10

    ealstm = MyEALSTM(input_size_dyn=input_size_dyn,
                    input_size_stat=input_size_stat,
                    hidden_size=n_hidden,
                    
                    initial_forget_bias= 5,
                    dropout= 0.4,
                    concat_static=False,
                    no_static=False)
    load_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/ea_lstm/group_{cluster_id}/pre_train/lambda=0/"
    load_model_name = f"group_{cluster_id}_ea_lstm_model_pre_train_{len(COLUNMNS_USE)}_{model_index}_lambda-0_train_on_obs"
    load_path = load_path_main+ load_model_name
    pretrain_dict = torch.load(load_path)['state_dict']
    model_dict = ealstm.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    ealstm.load_state_dict(pretrain_dict)

    #tell model to use GPU if needed
    if use_gpu:
        ealstm = ealstm.cuda()

    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(ealstm.parameters(), lr=learning_rate)

    # convergence variables
    converged = False

    training_errors = []
    validation_errors = []

    manualSeed = [random.randint(1, 99999999) for i in range(train_epochs)]
    for epoch in range(train_epochs):
        if verbose:
            print("train epoch: ", epoch)
        if use_gpu:
            set_seed(manualSeed[epoch])
        running_loss = 0.0

        # val_batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_val_batches)
        # valloader = DataLoader(val_data, batch_sampler=val_batch_sampler, pin_memory=True)

        batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)


        trainloader = DataLoader(train_data, batch_sampler=batch_sampler, pin_memory=True)

        #zero the parameter gradients
        optimizer.zero_grad()
        ealstm.train(True)
        avg_loss = 0
        batches_done = 0
        for i, b in enumerate(trainloader):

            #load data
            inputs = None
            targets = None

            unsup_inputs = None
            flux_data = None
            inputs, targets, Flux_data = b

            dynamic_inputs = torch.cat((inputs[:, :, :29], inputs[:, :, 39:]), dim=2)
            static_inputs = inputs[:, 0, 29:39]
            if(use_gpu):
                targets = targets.cuda()
                dynamic_inputs = dynamic_inputs.cuda()
                static_inputs = static_inputs.cuda()

            #forward  prop
            # ealstm.hidden = ealstm.init_hidden(batch_size=dynamic_inputs.size()[0])
            # h_state = None
            # print("static_inputs shape:", static_inputs.shape)
            outputs = ealstm(dynamic_inputs, static_inputs)[0]
            outputs = outputs.view(outputs.size()[0],-1)

            #calculate supervised loss
            loss_outputs = outputs[:,begin_loss_ind:]
            loss_targets = targets[:,begin_loss_ind:]            
            # open when training LSTM model with obs
            # loss_targets = obs_target[:,begin_loss_ind:]
                
            #get indices to calculate loss
            loss_indices = np.array(np.isfinite(loss_targets.cpu()), dtype='bool_')

            if use_gpu:
                loss_outputs = loss_outputs.cuda()
                loss_targets = loss_targets.cuda()


            supervised_loss = mse_criterion(loss_outputs[loss_indices], loss_targets[loss_indices])
            # R1 loss 
            if lambda_r1 > 0:
                reg1_loss = calculate_l1_loss(ealstm)
            else:
                reg1_loss = 0

            if torch.isnan(loss_targets).all():
                continue
                # print("loss_targets中全部都是NaN")

            # total loss
            loss = mse_lambda*supervised_loss + lambda_r1*reg1_loss 
            # loss = mse_lambda*supervised_loss + lambda_r1*reg1_loss + 0 * total_DOC_conservation_loss + 0 * stratified_DOC_conservation_loss
            #backward
            loss.backward(retain_graph=False)
            if grad_clip > 0:
                clip_grad_norm_(ealstm.parameters(), grad_clip, norm_type=2)

            #optimize
            optimizer.step()
            #zero the parameter gradients
            optimizer.zero_grad()
            avg_loss += loss
            batches_done += 1
            # print(f"epoch:{epoch},batch:{i}, un_loss = :{unsup_loss}")
        
        avg_loss = avg_loss / batches_done
        training_errors.append(avg_loss.cpu().item())
        if verbose:
            print("Supervised: ")
            print("Training loss=", avg_loss)
            print("=======================")

        if avg_loss < 1:
            if verbose:
                print("training converged")
            converged = True

        if converged:
            saveModel(ealstm.state_dict(), optimizer.state_dict(), save_path)
            print("training finished in ", epoch)
            break
        # break
    print("TRAINING COMPLETE")
    saveModel(ealstm.state_dict(), optimizer.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

def train_finetune_Transformer(lake_id, group_id, seed, model_index):
    COLUNMNS_USE = USE_FEATURES_COLUMNS_LAYER
    seq_length = 364
    n_hidden = 50
    win_shift = 364

    begin_loss_ind = 0#index in sequence where we begin to calculate error or predict
    grad_clip = 1.0 #how much to clip the gradient 2-norm in training
    yhat_batch_size = 2
    # hyper parameters
    ########################## UNFIXED ##########################
    use_gpu = True
    verbose = True
    train_epochs = 7
    n_features = len(COLUNMNS_USE) + 4
    doc_threshold = 0.5
    use_obs = True
    doc_lambda = 0
    mse_lambda = 1
    learning_rate = .002
    batch_size_ = 8

    model_type = 'transformer'
    model_status = 'fine_tune'
    lambda_r1 = 0 #magnitude hyperparameter of l1 loss

    cluster_id = group_id
    model_index = model_index

    save_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/{model_type}/group_{cluster_id}/{model_status}/lambda={doc_lambda}/individual_train_on_obs/"
    if not os.path.exists(save_path_main): 
        os.makedirs(save_path_main)

    model_name = f"{lake_id}_transformer_index_{model_index}_fine_tune_train_on_obs"
    save_path = save_path_main+ model_name
    print("save path:",save_path)

    ###############################
    # data preprocess
    ##################################
    #create train and test sets

    data_dir = f"../../data/processed/" # 
    ids = [lake_id]

    (train_data, trn_dates, val_data, val_dates, tst_data, tst_dates)  = buildManyLakeDataByIds(ids, data_dir, seq_length, n_features, win_shift, use_obs, evaluate=False)

    print("Training shape",train_data.shape)
    print("Validation shape",val_data.shape)
    print("Testing shape",tst_data.shape)
    print("=============")

    trn_data = train_data
    batch_size = batch_size_
    n_batches =  math.floor(trn_data.size()[0] / batch_size) 
    train_data = OxygenTrainDataset(trn_data)
    val_data = OxygenTrainDataset(val_data)
    print("n_batches:", n_batches)
    n_depths = 2
    yhat_batch_size = n_depths
    batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)
    print("Tranining batch size:", batch_size)
    num_heads = 5
    num_layers = 2
    hidden_size = 32
    model = TransformerModel(input_size=n_features, num_heads=num_heads, num_layers=num_layers, hidden_size=hidden_size)
    load_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/transformer/group_{cluster_id}/pre_train/lambda=0/"
    load_model_name = f"group_{cluster_id}_transformer_model_pre_train_{len(COLUNMNS_USE)}_{model_index}_lambda-0_train_on_obs"
    load_path = load_path_main+ load_model_name
    pretrain_dict = torch.load(load_path)['state_dict']
    model_dict = model.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    model.load_state_dict(pretrain_dict)

    #tell model to use GPU if needed
    if use_gpu:
        model = model.cuda()

    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # convergence variables
    converged = False

    training_errors = []
    validation_errors = []

    manualSeed = [random.randint(1, 99999999) for i in range(train_epochs)]
    for epoch in range(train_epochs):
        if verbose:
            print("train epoch: ", epoch)
        if use_gpu:
            set_seed(manualSeed[epoch])
        running_loss = 0.0
        batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_batches)
        trainloader = DataLoader(train_data, batch_sampler=batch_sampler, pin_memory=True)
        optimizer.zero_grad()
        model.train(True)
        avg_loss = 0
        batches_done = 0
        for i, b in enumerate(trainloader):
            inputs, targets, Flux_data = b

            if use_gpu:
                inputs = inputs.cuda()
                targets = targets.cuda()

            outputs = model(inputs)
            outputs = outputs.view(outputs.size()[0], -1)
            loss_outputs = outputs[:, begin_loss_ind:]
            loss_targets = targets[:, begin_loss_ind:]
            loss_indices = np.array(np.isfinite(loss_targets.cpu()), dtype='bool_')

            if use_gpu:
                loss_outputs = loss_outputs.cuda()
                loss_targets = loss_targets.cuda()

            supervised_loss = mse_criterion(loss_outputs[loss_indices], loss_targets[loss_indices])
            reg1_loss = calculate_l1_loss(model) if lambda_r1 > 0 else 0
            loss = mse_lambda * supervised_loss + lambda_r1 * reg1_loss
            loss.backward(retain_graph=False)
            if grad_clip > 0:
                clip_grad_norm_(model.parameters(), grad_clip, norm_type=2)
            optimizer.step()
            optimizer.zero_grad()
            avg_loss += loss
            batches_done += 1

        avg_loss = avg_loss / batches_done
        training_errors.append(avg_loss.cpu().item())
        if verbose:
            print("Supervised: ")
            print("Training loss=", avg_loss)
            print("=======================")
        if avg_loss < 1:
            if verbose:
                print("training converged")
            converged = True
        if converged:
            saveModel(model.state_dict(), optimizer.state_dict(), save_path)
            print("training finished in ", epoch)
            break

    print("TRAINING COMPLETE")
    saveModel(model.state_dict(), optimizer.state_dict(), save_path)
    print(f"Model saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train LSTM Model")
    parser.add_argument('--lake_id', type=int, required=True,
                        help='Lake ID to train the model on')
    args = parser.parse_args()
    train_finetune_Lstm(args.lake_id)
