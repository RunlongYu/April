import argparse
import json
import pickle
import random
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
from models.pytorch_model_operations import saveModel
from data.pytorch_data_operations import buildOneLakeData, buildManyLakeDataByIds, calculate_total_DOC_conservation_loss, calculate_stratified_DOC_conservation_loss
from data.pytorch_data_operations import ContiguousBatchSampler, RandomContiguousBatchSampler, MultiLoader
from data.lakePreprocess import USE_FEATURES_COLUMNS, USE_FEATURES_COLUMNS_LAYER, USE_FEATURES_COLUMNS_NOFLUX, FLUX_COLUMNS
from utils.train_utils import MyLSTM, MyEALSTM, TransformerModel, set_seed, random_seed, PretrainAprilCfg, PretrainLSTMCfg, PretrainPrilCfg, TransformerConfig
import sys
import os
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import pytorch_data_operations
# import pytorch_model_operations
# sys.path.append(os.path.dirname(sys.path[0]))


# sys.path.append('../../data')
# sys.path.append('../data')
# sys.path.append('../../models')
# sys.path.append('../models')
# sys.path.append('../../utils')
# sys.path.append('../utils')


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


def pretrain_model(args):
    ###########
    # Globals #
    ###########
    ########################## FIXED ##########################
    seq_length = 364
    n_hidden = 50
    win_shift = 364

    begin_loss_ind = 0  # index in sequence where we begin to calculate error or predict
    grad_clip = 1.0  # how much to clip the gradient 2-norm in training
    yhat_batch_size = 2
    doc_lambda = 0
    mse_lambda = 1
    # hyper parameters
    IS_FINE_TUNE = False
    ########################## UNFIXED ##########################
    use_gpu = True
    verbose = True
    cluster_id = args.cluster_id
    model_index = args.model_index

    model_type = args.model_type
    if model_type == 'lstm':
        config = PretrainLSTMCfg(cluster_id)
        model_status = 'pre_train_LSTM'
    elif model_type == 'pril':
        config = PretrainPrilCfg(cluster_id)
        model_status = 'pre_train_PRIL'
    elif model_type == 'april':
        config = PretrainAprilCfg(cluster_id)
        model_status = 'pre_train_APRIL'

    # cluster_id = 1
    # model_index = 1
    IS_PGRNN = config.IS_PGRNN
    USE_EXTEND = config.USE_EXTEND
    if USE_EXTEND:
        use_extend = '_use_extend'
    else:
        use_extend = ''

    train_epochs = config.train_epochs
    seed = random_seed[model_index-1]
    set_seed(seed)
    COLUNMNS_USE = USE_FEATURES_COLUMNS_LAYER
    n_features = len(COLUNMNS_USE) + 4
    n_flux = len(FLUX_COLUMNS)
    FLUX_START = -1-len(FLUX_COLUMNS)
    doc_threshold = 0

    use_obs = True
    mse_lambda = 1
    learning_rate = config.learning_rate
    batch_size_ = config.batch_size
    lambda_total = config.lambda_total
    lambda_stratified_epi = config.lambda_stratified_epi
    lambda_stratified_hypo = config.lambda_stratified_hypo

    # train April pre-train model
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

    if IS_FINE_TUNE:
        lambda_r1 = 0.00001  # magnitude hyperparameter of l1 loss
    else:
        model_type = 'pre_train'
        lambda_r1 = 0  # magnitude hyperparameter of l1 loss

    save_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/{model_type}/group_{cluster_id}/{model_status}/lambda={doc_lambda}/"
    if not os.path.exists(save_path_main):
        os.makedirs(save_path_main)

    model_name = f"group_{cluster_id}_{model_type}_model_{model_status}_{len(COLUNMNS_USE)}_{model_index}_lambda-{doc_lambda}{use_extend}_train_on_obs"
    save_path = save_path_main + model_name
    load_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/pre_train/group_{cluster_id}/pre_train/lambda=0/"
    load_model_name = f"group_{cluster_id}_pre_train_model_pre_train_{len(COLUNMNS_USE)}_{model_index}_lambda-0"
    load_path = load_path_main + load_model_name
    print("load path:", load_path)
    print("save path:", save_path)
    print("colunmns_use size:", len(COLUNMNS_USE))

    if not USE_EXTEND:
        data_dir = f"../../data/processed/"
        ids = pd.read_csv(
            f'../../data/utils/groups/vol_area/cluster_{cluster_id}.csv')
    else:
        data_dir = f"../../data/processed_extend/seed={seed}/"
        ids = pd.read_csv(
            f'../../data/utils/groups/vol_area/cluster_{cluster_id}.csv')

    ids = ids['nhdhr_id'].to_list()
    print("data_dir:", data_dir)

    (train_data, trn_dates, val_data, val_dates, tst_data, tst_dates) = buildManyLakeDataByIds(
        ids, data_dir, seq_length, n_features, win_shift, use_obs, evaluate=False)

    print("Training shape", train_data.shape)
    print("Validation shape", val_data.shape)
    print("Testing shape", tst_data.shape)
    print("=============")
    # if not IS_FINE_TUNE:
    #     trn_data = torch.cat((train_data, val_data), dim=0)
    #     trn_data = torch.cat((trn_data, tst_data), dim=0)
    #     print("After concatenate Test and Train:")
    #     print(trn_data.shape)
    # else:
    #     trn_data = torch.cat((train_data, val_data), dim=0)
    trn_data = train_data

    # unsup_data = torch.cat((train_data, val_data), dim=0)
    # unsup_data = torch.cat((unsup_data, tst_data), dim=0)
    unsup_data = tst_data

    print("unsup_data shape:")
    print(unsup_data.shape)

    unsup_data_size = unsup_data.shape[0]
    batch_size = batch_size_

    n_batches = math.floor(trn_data.size()[0] / batch_size)
    n_val_batches = math.floor(val_data.size()[0] / batch_size)
    n_batches_unsup = math.floor(unsup_data.size()[0] / batch_size)
    train_data = OxygenTrainDataset(trn_data)
    val_data = OxygenTrainDataset(val_data)
    unsup_data = OxygenTrainDataset(unsup_data)

    print("n_batches:", n_batches)
    print("n_batches_unsup", n_batches_unsup)
    n_depths = 2
    yhat_batch_size = n_depths

    batch_sampler = ContiguousBatchSampler(
        batch_size, n_batches)
    print("Tranining batch size:", batch_size)

    lstm_net = MyLSTM(n_features, n_hidden, batch_size, use_gpu)

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
    total_rmse_list = []
    lower_rmse_list = []
    manualSeed = [random.randint(1, 99999999) for i in range(train_epochs)]
    for epoch in range(train_epochs):
        if verbose:
            print("train epoch: ", epoch)
        if use_gpu:
            set_seed(manualSeed[epoch])
        running_loss = 0.0

        val_batch_sampler = ContiguousBatchSampler(
            batch_size, n_val_batches)
        valloader = DataLoader(
            val_data, batch_sampler=val_batch_sampler, pin_memory=True)

        batch_sampler = ContiguousBatchSampler(
            batch_size, n_batches)
        batch_sampler_unsup = RandomContiguousBatchSampler(
            unsup_data_size, 2, n_batches)  # yhat_batch_size = layer number

        # print("length batch_sampler:", len(batch_sampler))
        # print("length batch_sampler_unsup:", len(batch_sampler_unsup))

        unsupdataloader = DataLoader(
            unsup_data, batch_sampler=batch_sampler_unsup, pin_memory=True)
        trainloader = DataLoader(
            train_data, batch_sampler=batch_sampler, pin_memory=True)
        multi_loader = MultiLoader(
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

            # nan_count = torch.isnan(targets).sum().item()
            # non_nan_count = (~torch.isnan(targets)).sum().item()

            # # 打印结果
            # print(f"Number of NaN values: {nan_count}")
            # print(f"Number of non-NaN values: {non_nan_count}")

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

            # get unsupervised outputs
            unsup_outputs, h_state = lstm_net(unsup_inputs, h_state)

            if use_gpu:
                unsup_outputs = unsup_outputs.cuda()

            # calculate unsupervised loss
            if IS_PGRNN:
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
                upper_DOC_conservation_loss = 0
                lower_DOC_conservation_loss = 0

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

            # print("nan in target:", torch.sum(torch.isnan(loss_targets)))
            # print("nan in target:", torch.sum(~torch.isnan(loss_targets)))
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
        #     for inputs, targets, flux_data in valloader:

        #         if use_gpu:
        #             inputs, targets = inputs.cuda(), targets.cuda()

        #         loss_targets = targets

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
            print("---------------")
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


def pretrain_ealstm(args):
    model_index = args.model_index
    cluster_id =args.cluster_id
    seed = random_seed[model_index - 1]
    set_seed(seed)
    ###########
    # Globals #
    ###########
    ########################## FIXED ##########################
    IS_BUCKET = False
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

    train_epochs = 7
    n_features = len(COLUNMNS_USE) + 4
    n_flux = len(FLUX_COLUMNS)
    FLUX_START = -1-len(FLUX_COLUMNS)
    doc_threshold = 0.5

    use_obs = True
    doc_lambda = 0
    mse_lambda = 1
    learning_rate = .009
    batch_size_ = 32

    model_type = 'ea_lstm'
    model_status = 'pre_train'
    lambda_r1 = 0 #magnitude hyperparameter of l1 loss

    save_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/{model_type}/group_{cluster_id}/{model_status}/lambda={doc_lambda}/"
    if not os.path.exists(save_path_main): 
        os.makedirs(save_path_main)

    model_name = f"group_{cluster_id}_{model_type}_model_{model_status}_{len(COLUNMNS_USE)}_{model_index}_lambda-{doc_lambda}_train_on_obs"
    save_path = save_path_main+ model_name

    print("save path:",save_path)
    print("colunmns_use size:", len(COLUNMNS_USE))


    #Dataset classes
    class OxygenTrainDataset(Dataset):
        #training dataset class, allows Dataloader to load both input/target
        def __init__(self, trn_data):
            self.len = trn_data.shape[0]
            self.x_data = trn_data[:,:,:FLUX_START].float()
            self.flux_data = trn_data[:,:,FLUX_START:-1]
            self.y_target = trn_data[:,:,-1].float()

        def __getitem__(self, index):
            return self.x_data[index], self.y_target[index], self.flux_data[index]
        def __len__(self):
            return self.len

    # # pre-train
    ###############################
    # data preprocess
    ##################################
    #create train and test sets

    data_dir = f"../../data/processed/" # 
    ids = pd.read_csv(f'../../data/utils/groups/vol_area/cluster_{cluster_id}.csv')

    ids = ids['nhdhr_id'].to_list()


    (train_data, trn_dates, val_data, val_dates, tst_data, tst_dates)  = buildManyLakeDataByIds(ids, data_dir, seq_length, n_features, win_shift, use_obs, evaluate=False)

    print("Training shape",train_data.shape)
    print("Validation shape",val_data.shape)
    print("Testing shape",tst_data.shape)
    print("=============")

    trn_data = train_data
    batch_size = batch_size_
    n_batches =  math.floor(trn_data.size()[0] / batch_size) 
    n_val_batches = math.floor(val_data.size()[0] / batch_size) 
    train_data = OxygenTrainDataset(trn_data)
    val_data = OxygenTrainDataset(val_data)
    print("n_batches:", n_batches)
    n_depths = 2
    yhat_batch_size = n_depths
    batch_sampler = ContiguousBatchSampler(batch_size, n_batches)
    print("Tranining batch size:", batch_size)

    # %%
    input_size_dyn = 35
    input_size_stat = 10

    ealstm = MyEALSTM(input_size_dyn=input_size_dyn,
                    input_size_stat=input_size_stat,
                    hidden_size=n_hidden,
                    
                    initial_forget_bias= 5,
                    dropout= 0.4,
                    concat_static=False,
                    no_static=False)
    #tell model to use GPU if needed
    if use_gpu:
        ealstm = ealstm.cuda()


    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(ealstm.parameters(), lr=learning_rate)


    # convergence variables
    converged = False

    training_errors = []
    validation_errors = []

    # %%
    def calculate_l1_loss(model):
        def l1_loss(x):
            return torch.abs(x).sum()

        to_regularize = []
        for name, p in model.named_parameters():
            if 'bias' in name:
                continue 
            else:
                #take absolute value of weights and sum
                to_regularize.append(p.view(-1))
        # l1_loss_val = torch.tensor(1, requires_grad=True, dtype=torch.float32)
        l1_loss_val = l1_loss(torch.cat(to_regularize))
        return l1_loss_val


    # %%
    total_rmse_list = []
    lower_rmse_list = []
    manualSeed = [random.randint(1, 99999999) for i in range(train_epochs)]
    for epoch in range(train_epochs):
        if verbose:
            print("train epoch: ", epoch)
        if use_gpu:
            set_seed(manualSeed[epoch])
        running_loss = 0.0

        # val_batch_sampler = pytorch_data_operations.ContiguousBatchSampler(batch_size, n_val_batches)
        # valloader = DataLoader(val_data, batch_sampler=val_batch_sampler, pin_memory=True)

        batch_sampler = ContiguousBatchSampler(batch_size, n_batches)


        trainloader = DataLoader(train_data, batch_sampler=batch_sampler, pin_memory=True)

        #zero the parameter gradients
        optimizer.zero_grad()
        ealstm.train(True)
        avg_loss = 0
        avg_total_DOC_conservation_loss = 0
        avg_upper_DOC_conservation_loss = 0
        avg_lower_DOC_conservation_loss = 0
        batches_done = 0
        for i, b in enumerate(trainloader):

            #load data
            inputs = None
            targets = None

            unsup_inputs = None
            flux_data = None
            inputs, targets, Flux_data = b

            dynamic_inputs = torch.cat((inputs[:, :, :29], inputs[:, :, 39:]), dim=2)
            static_inputs = inputs[:, 0, 29:39]  # Assuming static inputs are the same for all timesteps

            # nan_count = torch.isnan(targets).sum().item()
            # non_nan_count = (~torch.isnan(targets)).sum().item()

            # # 打印结果
            # print(f"Number of NaN values: {nan_count}")
            # print(f"Number of non-NaN values: {non_nan_count}")

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

            # total_DOC_conservation_loss = calculate_total_DOC_conservation_loss(flux_data, unsup_outputs, doc_threshold, 1, use_gpu)
            # stratified_DOC_conservation_loss = calculate_stratified_DOC_conservation_loss(flux_data, unsup_outputs, doc_threshold, 1, use_gpu)

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

            # print("nan in target:", torch.sum(torch.isnan(loss_targets)))
            # print("nan in target:", torch.sum(~torch.isnan(loss_targets)))
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
    
        # test_total_rmse, test_lower_rmse = test_model(tst_data, ealstm)
        # total_rmse_list.append(test_total_rmse)
        # lower_rmse_list.append(test_lower_rmse)
        if verbose:
            print("Supervised: ")
            print("Training loss=", avg_loss)
            # print("Validation:")
            # print("Validation loss =",val_loss)
            print("=======================")

        if avg_loss < 1:
            if verbose:
                print("training converged")
            converged = True

        if converged:
            saveModel(ealstm.state_dict(), optimizer.state_dict(), save_path)
            print("training finished in ", epoch)
            break


def pretrain_transformer(args):
    transformer_cfg = TransformerConfig()
    model_index = args.model_index
    cluster_id =args.cluster_id
    seed = random_seed[model_index - 1]
    set_seed(seed)
    print("seed = ", seed)
    ###########
    # Globals #
    ###########
    ########################## FIXED ##########################
    IS_BUCKET = False
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

    train_epochs = 12
    n_features = len(COLUNMNS_USE) + 4
    n_flux = len(FLUX_COLUMNS)
    FLUX_START = -1-len(FLUX_COLUMNS)
    doc_threshold = 0.5

    use_obs = True
    doc_lambda = 0
    mse_lambda = 1
    learning_rate = transformer_cfg.learning_rate
    batch_size_ = transformer_cfg.batch_size

    model_type = 'transformer'
    model_status = 'pre_train'
    lambda_r1 = 0 #magnitude hyperparameter of l1 loss



    save_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/{model_type}/group_{cluster_id}/{model_status}/lambda={doc_lambda}/"
    if not os.path.exists(save_path_main): 
        os.makedirs(save_path_main)

    model_name = f"group_{cluster_id}_{model_type}_model_{model_status}_{len(COLUNMNS_USE)}_{model_index}_lambda-{doc_lambda}_train_on_obs"
    save_path = save_path_main+ model_name

    print("save path:",save_path)
    print("colunmns_use size:", len(COLUNMNS_USE))

    # %%
    #Dataset classes
    class OxygenTrainDataset(Dataset):
        #training dataset class, allows Dataloader to load both input/target
        def __init__(self, trn_data):
            self.len = trn_data.shape[0]
            self.x_data = trn_data[:,:,:FLUX_START].float()
            self.flux_data = trn_data[:,:,FLUX_START:-1]
            self.y_target = trn_data[:,:,-1].float()

        def __getitem__(self, index):
            return self.x_data[index], self.y_target[index], self.flux_data[index]
        def __len__(self):
            return self.len

    # %% [markdown]
    # # pre-train

    # %%
    ###############################
    # data preprocess
    ##################################
    #create train and test sets

    data_dir = f"../../data/processed/" # 
    ids = pd.read_csv(f'../../data/utils/groups/vol_area/cluster_{cluster_id}.csv')

    ids = ids['nhdhr_id'].to_list()


    (train_data, trn_dates, val_data, val_dates, tst_data, tst_dates)  = buildManyLakeDataByIds(ids, data_dir, seq_length, n_features, win_shift, use_obs, evaluate=False)

    print("Training shape",train_data.shape)
    print("Validation shape",val_data.shape)
    print("Testing shape",tst_data.shape)
    print("=============")

    trn_data = train_data
    batch_size = batch_size_
    n_batches =  math.floor(trn_data.size()[0] / batch_size) 
    n_val_batches = math.floor(val_data.size()[0] / batch_size) 
    train_data = OxygenTrainDataset(trn_data)
    val_data = OxygenTrainDataset(val_data)
    print("n_batches:", n_batches)
    n_depths = 2
    yhat_batch_size = n_depths
    batch_sampler = ContiguousBatchSampler(batch_size, n_batches)
    print("Tranining batch size:", batch_size)


    num_heads = transformer_cfg.num_heads
    num_layers = transformer_cfg.num_layers
    hidden_size = transformer_cfg.hidden_size

    model = TransformerModel(input_size=n_features, num_heads=num_heads, num_layers=num_layers, hidden_size=hidden_size)

    #tell model to use GPU if needed
    if use_gpu:
        model = model.cuda()


    mse_criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)


    # convergence variables
    converged = False

    training_errors = []
    validation_errors = []

    # %%
    def calculate_l1_loss(model):
        def l1_loss(x):
            return torch.abs(x).sum()

        to_regularize = []
        for name, p in model.named_parameters():
            if 'bias' in name:
                continue 
            else:
                #take absolute value of weights and sum
                to_regularize.append(p.view(-1))
        # l1_loss_val = torch.tensor(1, requires_grad=True, dtype=torch.float32)
        l1_loss_val = l1_loss(torch.cat(to_regularize))
        return l1_loss_val
        
    # %%
    total_rmse_list = []
    lower_rmse_list = []
    manualSeed = [random.randint(1, 99999999) for i in range(train_epochs)]
    for epoch in range(train_epochs):
        if verbose:
            print("train epoch: ", epoch)
        if use_gpu:
            set_seed(manualSeed[epoch])
        running_loss = 0.0
        batch_sampler = ContiguousBatchSampler(batch_size, n_batches)
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

    # %%
    ids = pd.read_csv(f'../../data/utils/groups/vol_area/removed/cluster_{cluster_id}.csv')  
    ids = ids['nhdhr_id'].to_list()

    load_path_main = f"../../models/{len(COLUNMNS_USE)}/seed={seed}/transformer/group_{cluster_id}/pre_train/lambda=0/"
    load_model_name = f"group_{cluster_id}_transformer_model_pre_train_{len(COLUNMNS_USE)}_{model_index}_lambda-0_train_on_obs"
    load_path = load_path_main+ load_model_name
    print("load path:", load_path)