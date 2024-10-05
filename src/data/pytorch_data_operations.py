import torch
import numpy as np
import pandas as pd
import math
import sys
import datetime
from datetime import date
import os
import random
from tqdm import tqdm
import torch.nn as nn
from torch.nn.init import xavier_normal_
import pdb
from scipy import interpolate


def getBatch(data, dates, seq_length, win_shift):
    n_layer, n_dates, n_features = data.shape  # for example: [2, 1000. 42]
    seq_per_layer = math.floor(n_dates / seq_length)

    seq_per_layer = (n_dates - seq_length) // win_shift + \
        1    # numer of batches
    n_seq = seq_per_layer * n_layer
    Data_batch = np.empty(
        shape=(n_seq, seq_length, n_features))  # features + label
    Dates_batch = np.empty(shape=(n_seq, seq_length), dtype='datetime64[s]')
    Data_batch[:] = np.nan
    Dates_batch[:] = np.datetime64("NaT")
    ct = 0
    for seq in range(seq_per_layer):
        for l in range(n_layer):
            start_idx = seq * win_shift
            end_idx = start_idx + seq_length
            Data_batch[ct, :, :] = data[l, start_idx:end_idx, :]
            Dates_batch[ct, :] = dates[start_idx:end_idx]
            ct += 1

    valid_indices = []
    for seq in range(0, n_seq, n_layer):  # step by n_layer to check pairs
        pair_has_nan = False
        for l in range(n_layer):
            obs_target = Data_batch[seq + l, :, -1]
            if np.isnan(obs_target).all():
                pair_has_nan = True
                break
        if not pair_has_nan:
            valid_indices.extend([seq, seq + 1])

    Data_batch = Data_batch[valid_indices, :, :]
    Dates_batch = Dates_batch[valid_indices, :]

    for i in range(Data_batch.shape[0]):
        assert not np.isnan(Data_batch[i]).all(
        ), f"Row {i} in Data_batch is all NaN"

    return torch.from_numpy(Data_batch), Dates_batch


def buildOneLakeData(lakename, data_dir, seq_length, n_features, win_shift, use_obs, evaluate):
    debug = False
    my_path = os.path.abspath(os.path.dirname(__file__))
    my_path = ""
    # GET TRAIN/VAL/TEST HERE
    if not evaluate:
        if not use_obs:
            trn_data = np.load(os.path.join(
                my_path, data_dir+lakename+"/trn_pre_train_data.npy"))
            val_data = np.load(os.path.join(
                my_path, data_dir+lakename+"/val_pre_train_data.npy"))
            tst_data = np.load(os.path.join(
                my_path, data_dir+lakename+"/tst_pre_train_data.npy"))
        else:
            trn_data = np.load(os.path.join(
                my_path, data_dir+lakename+"/trn_fine_tune_data.npy"))
            val_data = np.load(os.path.join(
                my_path, data_dir+lakename+"/val_fine_tune_data.npy"))
            tst_data = np.load(os.path.join(
                my_path, data_dir+lakename+"/tst_fine_tune_data.npy"))
    else:
        trn_data = np.load(os.path.join(my_path, data_dir +
                           lakename+"/trn_evaluate_data.npy"))
        val_data = np.load(os.path.join(my_path, data_dir +
                           lakename+"/val_evaluate_data.npy"))
        tst_data = np.load(os.path.join(my_path, data_dir +
                           lakename+"/tst_evaluate_data.npy"))

    # layer = trn_data[:, :, -1:]
    # for i in range(4):
    #     trn_data = np.concatenate((trn_data, layer), axis=2)

    # # For val_data
    # layer = val_data[:, :, -1:]
    # for i in range(4):
    #     val_data = np.concatenate((val_data, layer), axis=2)

    # # For tst_data
    # layer = tst_data[:, :, -1:]
    # for i in range(4):
    #     tst_data = np.concatenate((tst_data, layer), axis=2)

    trn_dates = np.load(os.path.join(
        my_path, data_dir+lakename+"/trn_dates.npy"))
    val_dates = np.load(os.path.join(
        my_path, data_dir+lakename+"/val_dates.npy"))
    tst_dates = np.load(os.path.join(
        my_path, data_dir+lakename+"/tst_dates.npy"))

    Train_Data, Train_Dates = getBatch(
        trn_data, trn_dates, seq_length, win_shift)
    Val_Data, Val_Dates = getBatch(val_data, val_dates, seq_length, win_shift)
    Test_Data, Test_Dates = getBatch(
        tst_data, tst_dates, seq_length, win_shift)

    return (Train_Data, Train_Dates, Val_Data, Val_Dates, Test_Data, Test_Dates)


def buildManyLakeDataByIds(ids, data_dir, seq_length, n_features, win_shift, use_obs, evaluate):
    all_trn_data = []
    all_trn_dates = []
    all_val_data = []
    all_val_dates = []
    all_tst_data = []
    all_tst_dates = []
    for count, lake_id in enumerate(ids):
        (trn_data, trn_dates, val_data, val_dates, tst_data, tst_dates) = buildOneLakeData(
            lake_id, data_dir, seq_length, n_features, win_shift, use_obs, evaluate)

        all_trn_data.append(trn_data)
        all_trn_dates.append(trn_dates)
        all_val_data.append(val_data)
        all_val_dates.append(val_dates)
        all_tst_data.append(tst_data)
        all_tst_dates.append(tst_dates)

    Train_Data = torch.cat(all_trn_data, dim=0)
    Val_Data = torch.cat(all_val_data, dim=0)
    Test_Data = torch.cat(all_tst_data, dim=0)

    Train_Dates = np.concatenate(all_trn_dates)
    Val_Dates = np.concatenate(all_val_dates)
    Test_Dates = np.concatenate(all_tst_dates)

    return (Train_Data, Train_Dates, Val_Data, Val_Dates, Test_Data, Test_Dates)


def buildOneLakeData_ForTFT(group_id, lakename, seed, data_dir, seq_length, n_features, win_shift, use_obs, evaluate):
    debug = False
    my_path = os.path.abspath(os.path.dirname(__file__))
    if_use_tft = False
    # GET TRAIN/VAL/TEST HERE
    if not evaluate:
        if not use_obs:
            trn_data = np.load(os.path.join(
                my_path, data_dir+lakename+"/trn_pre_train_data.npy"))
            val_data = np.load(os.path.join(
                my_path, data_dir+lakename+"/val_pre_train_data.npy"))
            tst_data = np.load(os.path.join(
                my_path, data_dir+lakename+"/tst_pre_train_data.npy"))
        else:
            trn_data = np.load(os.path.join(
                my_path, data_dir+lakename+"/trn_fine_tune_data.npy"))
            val_data = np.load(os.path.join(
                my_path, data_dir+lakename+"/val_fine_tune_data.npy"))
            tst_data = np.load(os.path.join(
                my_path, data_dir+lakename+"/tst_fine_tune_data.npy"))
    else:
        trn_data = np.load(os.path.join(my_path, data_dir +
                           lakename+"/trn_evaluate_data.npy"))
        val_data = np.load(os.path.join(my_path, data_dir +
                           lakename+"/val_evaluate_data.npy"))
        tst_data = np.load(os.path.join(my_path, data_dir +
                           lakename+"/tst_evaluate_data.npy"))

        tft_epi_pred_path = f'../../results/data/seed={seed}/tft/cluster_{group_id}/{lakename}.csv'
        # TO-DO change TFT to tft_hypo
        tft_hypo_pred_path = f'../../results/data/seed={seed}/tft_hypo/cluster_{group_id}/{lakename}.csv'

        tft_epi_pred = pd.read_csv(tft_epi_pred_path, usecols=['Prediction'])
        tft_hypo_pred = pd.read_csv(tft_hypo_pred_path, usecols=['Prediction'])
        tft_epi_pred_value = tft_epi_pred.to_numpy()
        tft_hypo_pred_value = tft_hypo_pred.to_numpy()

        print("tst_data[0,:,-2] shape:", tst_data[0, :, -2].shape)
        print("tft_epi_pred_value shape:", tft_epi_pred_value.shape)
        tst_data[0, :, -2] = np.squeeze(tft_epi_pred_value)
        tst_data[1, :, -2] = np.squeeze(tft_hypo_pred_value)
        if_use_tft = True

    if if_use_tft:
        print("Convert sim to tft results complete")
    # layer = trn_data[:, :, -1:]
    # for i in range(4):
    #     trn_data = np.concatenate((trn_data, layer), axis=2)

    # # For val_data
    # layer = val_data[:, :, -1:]
    # for i in range(4):
    #     val_data = np.concatenate((val_data, layer), axis=2)

    # # For tst_data
    # layer = tst_data[:, :, -1:]
    # for i in range(4):
    #     tst_data = np.concatenate((tst_data, layer), axis=2)

    trn_dates = np.load(os.path.join(
        my_path, data_dir+lakename+"/trn_dates.npy"))
    val_dates = np.load(os.path.join(
        my_path, data_dir+lakename+"/val_dates.npy"))
    tst_dates = np.load(os.path.join(
        my_path, data_dir+lakename+"/tst_dates.npy"))

    Train_Data, Train_Dates = getBatch(
        trn_data, trn_dates, seq_length, win_shift)
    Val_Data, Val_Dates = getBatch(val_data, val_dates, seq_length, win_shift)
    Test_Data, Test_Dates = getBatch(
        tst_data, tst_dates, seq_length, win_shift)

    return (Train_Data, Train_Dates, Val_Data, Val_Dates, Test_Data, Test_Dates)


def buildManyLakeDataByIds_ForTFT(group_id, ids, seed, data_dir, seq_length, n_features, win_shift, use_obs, evaluate):
    all_trn_data = []
    all_trn_dates = []
    all_val_data = []
    all_val_dates = []
    all_tst_data = []
    all_tst_dates = []
    for count, lake_id in tqdm(enumerate(ids)):
        (trn_data, trn_dates, val_data, val_dates, tst_data, tst_dates) = buildOneLakeData_ForTFT(
            group_id, lake_id, seed, data_dir, seq_length, n_features, win_shift, use_obs, evaluate)

        all_trn_data.append(trn_data)
        all_trn_dates.append(trn_dates)
        all_val_data.append(val_data)
        all_val_dates.append(val_dates)
        all_tst_data.append(tst_data)
        all_tst_dates.append(tst_dates)

    Train_Data = torch.cat(all_trn_data, dim=0)
    Val_Data = torch.cat(all_val_data, dim=0)
    Test_Data = torch.cat(all_tst_data, dim=0)

    Train_Dates = np.concatenate(all_trn_dates)
    Val_Dates = np.concatenate(all_val_dates)
    Test_Dates = np.concatenate(all_tst_dates)

    return (Train_Data, Train_Dates, Val_Data, Val_Dates, Test_Data, Test_Dates)


def calculate_stratified_flux(flux_data, t):
    # print("flux shape:", flux_data.shape)

    flux = flux_data.squeeze(0)
    fnep = flux[:, 0]  # simulated net ecosystem production flux
    fmineral = flux[:, 1]  # simulated mineralisation flux( hypo NEP)
    fsed = flux[:, 2]  # simulated net sedimentation flux
    fatm = flux[:, 3]  # simulated atmospheric exchange flux
    fdiff = flux[:, 4]  # simulated diffusion flux
    fentr_epi = flux[:, 5]  # simulated entrainment flux (epilimnion)
    fentr_hyp = flux[:, 6]  # simulated entrainment flux (hypolimnion)

    Flux_epi = fatm + fnep + fdiff  # F_ATM + F_NEP_epi + F_ENTR_epi + F_DIFF_epi
    Flux_hypo = fmineral - fsed - fdiff

    result = torch.cat(
        (Flux_epi.unsqueeze(0), Flux_hypo.unsqueeze(0)), dim=0)  # size: [2,350]
    result = result * 0.001

    return result  # [2, 350]


def calculate_total_flux(flux_data, V_epi, V_hypo, t):
    # print("flux shape:", flux_data.shape)

    flux = flux_data.squeeze(0)
    fnep = flux[:, 0]  # simulated net ecosystem production flux
    fmineral = flux[:, 1]  # simulated mineralisation flux( hypo NEP)
    fsed = flux[:, 2]  # simulated net sedimentation flux
    fatm = flux[:, 3]  # simulated atmospheric exchange flux
    fdiff = flux[:, 4]  # simulated diffusion flux
    fentr_epi = flux[:, 5]  # simulated entrainment flux (epilimnion)
    fentr_hyp = flux[:, 6]  # simulated entrainment flux (hypolimnion)

    mixed_mask = V_hypo == 0

    V_total = V_epi[0] + V_hypo[0]
    Flux_total = torch.zeros_like(fatm)

    # mixed
    Flux_total[mixed_mask] = fatm[mixed_mask] + \
        fnep[mixed_mask] + fmineral[mixed_mask] - fsed[mixed_mask]

    # stratified
    # flux_epi =  fatm + fnep + fentr_epi + fdiff # F_ATM + F_NEP_epi + F_ENTR_epi + F_DIFF
    # flux_hypo = fmineral - fsed - fentr_hyp - fdiff # F_NEP_hypo - F_SED + F_ENTR_hypo - F_DIFF
    # Flux_total[~mixed_mask] = (flux_epi[~mixed_mask] * V_epi[~mixed_mask] + flux_hypo[~mixed_mask] * V_hypo[~mixed_mask]) / V_total

    # print("result shape:", result.shape)
    Flux_total = Flux_total * 0.001
    return Flux_total  # [350]


def get_Flux_enter(flux_data, t):
    flux = flux_data.squeeze(0)
    fentr_epi = flux[:, 5]  # simulated entrainment flux (epilimnion)
    fentr_hyp = flux[:, 6]  # simulated entrainment flux (hypolimnion)

    Flux_enter = torch.cat((fentr_epi.unsqueeze(
        0), fentr_hyp.unsqueeze(0)), dim=0)  # size: [2,350]
    Flux_enter = Flux_enter*0.001
    return Flux_enter


def find_stratified_segments(tensor):
    stratified_segments = []
    start = None

    for i, val in enumerate(tensor):
        if val != 0 and start is None:
            start = i
        elif val == 0 and start is not None:
            stratified_segments.append((start, i))
            start = None

    if start is not None:
        stratified_segments.append((start, len(tensor)))

    return stratified_segments


def get_extend_seg_loss_old(Do_next, Do_pre, V_both_next, V_both_pre, flux, extend_info):
    torch.set_printoptions(precision=8) 
    debug = False
    k = 12  # Hyperparameter
    seq_length = Do_next.shape[1]
    Delta_V = V_both_next - V_both_pre
    delta_v = Delta_V/k

    delta_flux_const = flux/k
    new_Do_next = Do_pre.clone()
    old_Do_pre = Do_pre.clone()

    if debug:
        # ex: flux data shape: torch.Size([2, 152])
        print("flux data shape:", delta_flux_const.shape)
        print("Do_pre", Do_pre.shape)
    for i in range(k):
        if debug:
            print("i = ", i)
        for date in range(seq_length-1):
            if extend_info[date] == 1:
                if debug:
                    print("V_both_pre[:, date] epi:", V_both_pre[0, date])
                    print("Do_pre[:, date] epi before:", old_Do_pre[0, date])
                v_pre = V_both_pre[:, date] + delta_v[:, date]*(i)
                v_next = V_both_pre[:, date] + delta_v[:, date]*(i + 1)

                do = 0
                if delta_v[0, date] > 0:  # volume of upper layer increase
                    do = old_Do_pre[1, date]
                else:                    # volume of upper layer decrease
                    do = old_Do_pre[0, date]
                new_Do_next[:, date] = torch.maximum(
                    (old_Do_pre[:, date] * v_pre + delta_flux_const[:, date] * V_both_pre[:, date] + delta_v[:, date] * do)/v_next, torch.tensor(0.0))

                if debug:
                    # print("Do_pre[:, date] epi before:", old_Do_next[:, date])
                    print()
                old_Do_pre[:, date] = new_Do_next[:, date]
                if debug:
                    print("Do_pre[:, date] epi after:", old_Do_pre[:, date])
                    print("----------------------")

    extend_seg_loss = Do_next - new_Do_next
    return extend_seg_loss


def get_extend_seg_loss(Do_next, Do_pre, V_both_next, V_both_pre, flux, extend_info, use_gpu):
    torch.set_printoptions(precision=8)  
    k = 12  # Hyperparameter
    seq_length = Do_next.shape[1]
    Delta_V = V_both_next - V_both_pre
    delta_v = Delta_V / k
    delta_flux_const = flux / k
    new_Do_next = Do_pre.clone()
    old_Do_pre = Do_pre.clone()
    # Create a mask for dates where extend_info is 1
    mask = extend_info == 1
    if use_gpu and torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'
    delta_v_i = delta_v.unsqueeze(
        0) * torch.arange(k, device=device).view(-1, 1, 1)
    delta_v_ip1 = delta_v.unsqueeze(
        0) * torch.arange(1, k + 1, device=device).view(-1, 1, 1)

    for i in range(k):
        v_pre = V_both_pre.unsqueeze(0) + delta_v_i[i]
        v_next = V_both_pre.unsqueeze(0) + delta_v_ip1[i]

        v_pre_masked = v_pre[:, :, mask]
        v_next_masked = v_next[:, :, mask]
        old_Do_pre_masked = old_Do_pre[:, mask]
        do_masked = torch.where(
            delta_v[0, mask] > 0, old_Do_pre[1, mask], old_Do_pre[0, mask])
        new_Do_next_masked = torch.maximum(
            (old_Do_pre_masked * v_pre_masked +
             delta_flux_const[:, mask] * V_both_pre[:, mask] + delta_v[:, mask] * do_masked) / v_next_masked,
            torch.tensor(0.0)
        ).to(dtype=torch.float32)
        old_Do_pre[:, mask] = new_Do_next_masked
        new_Do_next[:, mask] = new_Do_next_masked

    extend_seg_loss = Do_next - new_Do_next
    return extend_seg_loss


def calculate_loss_for_segment(segment, V_Both, Flux_Data, Extend_info, Pred_, t, use_gpu):
    debug = False
    Flux = calculate_stratified_flux(Flux_Data, t)
    Flux_enter = get_Flux_enter(Flux_Data, t)
    start = segment[0]
    end = segment[1]
    V_both = V_Both[:, start:end]
    pred = Pred_[:, start:end]
    flux = Flux[:, start:end]
    flux_enter = Flux_enter[:, start:end]
    extend_info = Extend_info[start:end]  # Extend_info shape torch.Size([364])
    extend_info = extend_info[1:]
    Do_pre = pred[:, :-1]
    Do_next = pred[:, 1:]
    flux = flux[:, 1:]
    if debug:
        print("Do_next:", Do_next[1::2, :])
    V_both_pre = V_both[:, :-1]
    V_both_next = V_both[:, 1:]

    assert torch.all(
        V_both != 0), "V_total contains zero, which will cause division by zero."

    V_divided = V_both_pre/V_both_next
    if debug:
        print("V_divided shape:", V_divided.shape)

    seg_loss = Do_next - \
        torch.maximum(((Do_pre+flux)*V_divided +
                      flux_enter[:, 1:]), torch.tensor(0.0))
    seg_loss = seg_loss.to(Do_pre.dtype)
    if debug:
        print("1 in extend_info: ", torch.sum(extend_info))
    if torch.sum(extend_info) > 0:
        extend_seg_loss = get_extend_seg_loss(
            Do_next, Do_pre, V_both_next, V_both_pre, flux, extend_info, use_gpu)
        extend_info_mask = extend_info.unsqueeze(0).expand_as(seg_loss).bool()
        seg_loss[extend_info_mask] = extend_seg_loss[extend_info_mask]
    if debug:
        print("seg_loss lower", seg_loss[1::2, :])
    return seg_loss


def calculate_stratified_DOC_conservation_loss_old(flux_data, pred, doc_threshold, t, use_gpu):
    debug = False
    Volumes = flux_data[:, :, :2]  # size: [2,350,2]
    V_epi = Volumes[0, :, 0]  # size: [350]
    V_hypo = Volumes[1, :, 1]  # size: [350]
    V_both = torch.cat(
        (V_epi.unsqueeze(0), V_hypo.unsqueeze(0)), dim=0)  # size: [2,350]

    stratified_segs = find_stratified_segments(V_hypo)
    Flux_Data = flux_data[0, :, -7:]
    Pred = pred.squeeze(2)  # size: [2, 350]
    time_threshold = 5

    stratified_DO_loss = torch.tensor([])

    for segment in stratified_segs:
        if segment[1] - segment[0] >= time_threshold:
            segment_loss = calculate_loss_for_segment(
                segment, V_both, Flux_Data, Pred, t)
            if stratified_DO_loss.numel() == 0:
                stratified_DO_loss = segment_loss
            else:
                stratified_DO_loss = torch.cat(
                    (stratified_DO_loss, segment_loss), dim=1)

    stratified_DO_loss = stratified_DO_loss.abs()
    # stratified_loss = torch.clamp(stratified_DO_loss - 0, min=0)
    mae_stratified_loss_upper = stratified_DO_loss[::2, :].mean()
    mae_stratified_loss_lower = stratified_DO_loss[1::2, :].mean()
    return mae_stratified_loss_upper, mae_stratified_loss_lower


def calculate_stratified_DOC_conservation_loss(flux_data, pred, doc_threshold, t, use_gpu):
    debug = False
    Volumes = flux_data[:, :, :2]  # size: [2,350,2]
    V_epi = Volumes[0, :, 0]  # size: [350]
    V_hypo = Volumes[1, :, 1]  # size: [350]
    V_both = torch.cat(
        (V_epi.unsqueeze(0), V_hypo.unsqueeze(0)), dim=0)  # size: [2,350]

    stratified_segs = find_stratified_segments(V_hypo)
    Flux_Data = flux_data[0, :, 2:9]
    extend_info = flux_data[0, :, -1]
    Pred = pred.squeeze(2)  # size: [2, 350]
    time_threshold = 0

    stratified_DO_loss = torch.tensor([])
    for segment in stratified_segs:
        if segment[1] - segment[0] >= time_threshold:
            segment_loss = calculate_loss_for_segment(
                segment, V_both, Flux_Data, extend_info, Pred, t, use_gpu)
            if stratified_DO_loss.numel() == 0:
                stratified_DO_loss = segment_loss
            else:
                stratified_DO_loss = torch.cat(
                    (stratified_DO_loss, segment_loss), dim=1)

    stratified_DO_loss = stratified_DO_loss.abs()
    # stratified_loss = torch.clamp(stratified_DO_loss - 0, min=0)
    mae_stratified_loss_upper = stratified_DO_loss[::2, :].mean()
    mae_stratified_loss_lower = stratified_DO_loss[1::2, :].mean()

    return mae_stratified_loss_upper, mae_stratified_loss_lower


def calculate_total_DOC_conservation_loss(flux_data, pred, doc_threshold, t, use_gpu):
    debug = False
    Volumes = flux_data[:, :, :2]  # size: [2,350,2]
    V_epi = Volumes[0, :, 0]  # size: [350]
    V_hypo = Volumes[1, :, 1]  # size: [350]
    # V_both = torch.cat((V_epi.unsqueeze(0), V_hypo.unsqueeze(0)), dim=0) # size: [2,350]
    V_total = V_epi[0] + V_hypo[0]  # For a lake, the volume is a constant.
    Mixed_mask = V_hypo == 0
    Mixed_mask = Mixed_mask[1:].unsqueeze(0)
    if debug:
        V_hypo = Volumes[1, :, 1]

    Flux_Data = flux_data[0, :, 2:9]
    Pred = pred.squeeze(2)  # size: [2, 350]

    assert not torch.isnan(V_total).any(), "loss has nan V_total!!!"
    assert not torch.isnan(Flux_Data).any(), "loss has nan Flux_Data!!!"

    Flux = calculate_total_flux(Flux_Data, V_epi, V_hypo, t)

    Do_pre = Pred[::2, :-1]
    Do_next = Pred[::2, 1:]

    Do_pred_flux = Do_next - Do_pre
    total_DO_loss = Do_pred_flux - Flux[1:]
    if debug:
        print("Do_pred_flux", Do_pred_flux)
        print("Flux", Flux[1:])
    total_DO_loss = torch.clamp(total_DO_loss - doc_threshold, min=0)
    assert not torch.isnan(total_DO_loss).any(
    ), "total_DO_loss contains NaN values"
    assert not torch.isnan(Mixed_mask).any(), "Mixed_mask contains NaN values"

    mae_tota_loss = torch.tensor(0.0, device='cuda' if use_gpu else 'cpu')

    if Mixed_mask.any():
        # Calculate mae_tota_loss
        mae_tota_loss = torch.mean(torch.abs(total_DO_loss[Mixed_mask]))
        # print("Warning: Mixed_mask has no True values. Setting mae_tota_loss to 0.")
    assert not torch.isnan(mae_tota_loss).any(
    ), "mae_tota_loss contains NaN values"

    return mae_tota_loss


def clip_grad_norm_(parameters, max_norm, norm_type=2):
    r"""Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Tensor] or Tensor): an iterable of Tensors or a
            single Tensor that will have gradients normalized
        max_norm (float or int): max norm of the gradients
        norm_type (float or int): type of the used p-norm. Can be ``'inf'`` for
            infinity norm.

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    if isinstance(parameters, torch.Tensor):
        parameters = [parameters]
    parameters = list(filter(lambda p: p.grad is not None, parameters))
    max_norm = float(max_norm)
    norm_type = float(norm_type)
    if norm_type == inf:
        total_norm = max(p.grad.data.abs().max() for p in parameters)
    else:
        total_norm = 0
        for p in parameters:
            param_norm = p.grad.data.norm(norm_type)
            total_norm += param_norm.item() ** norm_type
        total_norm = total_norm ** (1. / norm_type)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            p.grad.data.mul_(clip_coef)
    return total_norm

# define LSTM model class


class LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, use_gpu):
        super(LSTM, self).__init__()
        self.use_gpu = use_gpu
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        # batch_first=True?
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)  # 1?
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=0):
        # initialize both hidden layers
        if batch_size == 0:
            batch_size = self.batch_size
        ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
               xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
        # print("hidden layer initialized: ", ret)
        # if data_parallel: #TODO??
        #     ret = (xavier_normal_(torch.empty(1, math.ceil(self.batch_size/2), self.hidden_size/2)),
        #         xavier_normal_(torch.empty(1, math.floorself.batch_size, math.floor(self.hidden_size/2))))
        if self.use_gpu:
            item0 = ret[0].cuda(non_blocking=True)
            item1 = ret[1].cuda(non_blocking=True)
            ret = (item0, item1)
        return ret

    def init_hidden_test(self, batch_size=0):
        # initialize both hidden layers
        if batch_size == 0:
            batch_size = self.batch_size
        torch.manual_seed(0)
        if self.use_gpu:
            torch.cuda.manual_seed_all(0)
        # print("epoch ", epoch+1)
        ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
               xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
        # print("hidden layer initialized: ", ret)
        # if data_parallel: #TODO??
        #     ret = (xavier_normal_(torch.empty(1, math.ceil(self.batch_size/2), self.hidden_size/2)),
        #         xavier_normal_(torch.empty(1, math.floorself.batch_size, math.floor(self.hidden_size/2))))
        if self.use_gpu:
            item0 = ret[0].cuda(non_blocking=True)
            item1 = ret[1].cuda(non_blocking=True)
            ret = (item0, item1)
        return ret

    def forward(self, x, hidden):
        # print("X size is {}".format(x.size()))
        self.lstm.flatten_parameters()

        x = x.float()
        x, hidden = self.lstm(x, self.hidden)
        self.hidden = hidden
        x = self.out(x)
        return x, hidden


class ContiguousBatchSampler(object):
    def __init__(self, batch_size, n_batches):
        # print("batch size", batch_size)
        # print("n batch ", n_batches)
        self.sampler = torch.randperm(n_batches)
        self.batch_size = batch_size

    def __iter__(self):
        for idx in self.sampler:
            yield torch.arange(idx*self.batch_size, (idx+1)*self.batch_size, dtype=torch.long)

    def __len__(self):
        return len(self.sampler) // self.batch_size


class RandomContiguousBatchSampler(object):
    def __init__(self, n_dates, batch_size, n_batches):
        high = math.floor(n_dates/batch_size)

        self.sampler = torch.randint_like(
            torch.empty(n_batches), low=0, high=high)
        self.batch_size = batch_size

    def __iter__(self):
        for idx in self.sampler:
            # yield torch.arange(idx*self.batch_size, (idx+1)*self.batch_size, dtype=torch.long) #old
            yield torch.arange(idx*self.batch_size, (idx+1)*self.batch_size, dtype=torch.long)

    def __len__(self):
        return len(self.sampler) // self.batch_size


def parseMatricesFromSeqs(pred, targ, depths, dates, n_depths, n_tst_dates, u_depths, u_dates):
    # format an array of sequences into one [depths x timestep] matrix
    assert pred.shape[0] == targ.shape[0]
    n_seq = pred.shape[0]
    seq_len = int(pred.shape[1])
    out_mat = np.empty((n_depths, n_tst_dates))
    out_mat[:] = np.nan
    lab_mat = np.empty((n_depths, n_tst_dates))
    lab_mat[:] = np.nan
    for i in np.arange(n_seq-1, -1, -1):
        # for each sequence
        if i >= dates.shape[0]:
            print("more sequences than dates")
            continue
        # find depth index
        if np.isnan(depths[i, 0]):
            print("nan depth")
            continue
        depth_ind = np.where(abs(u_depths - depths[i, 0].item()) <= .001)[0][0]

        # find date index
        if np.isnat(dates[i, 0]):
            print("not a time found")
            continue
        if len(np.where(u_dates == dates[i, 0])[0]) == 0:
            print("invalid date")
            continue
        date_ind = np.where(u_dates == dates[i, 0])[0][0]
        # print("depth ind: ", depth_ind, ", date ind: ",date_ind)
        if out_mat[depth_ind, date_ind:].shape[0] < seq_len:
            # this is to not copy data beyond test dates
            sizeToCopy = out_mat[depth_ind, date_ind:].shape[0]
            out_mat[depth_ind, date_ind:] = pred[i, :sizeToCopy]
            lab_mat[depth_ind, date_ind:] = targ[i, :sizeToCopy]
        else:
            indices = np.isfinite(targ[i, :])
            out_mat[depth_ind, date_ind:date_ind+seq_len] = pred[i, :]
            lab_mat[depth_ind, date_ind:date_ind +
                    seq_len][indices] = targ[i, :][indices]
            # for t in range(seq_len):
            #     if np.isnan(out_mat[depth_ind,date_ind+t]):
            #         out_mat[depth_ind, date_ind+t] = pred[i,t]
            #         lab_mat[depth_ind, date_ind+t] = targ[i,t]
        # print(np.count_nonzero(np.isfinite(lab_mat))," labels set")

    return (out_mat, lab_mat)


def transformTempToDensity(temp, use_gpu):
    # print(temp)
    # converts temperature to density
    # parameter:
    # @temp: single value or array of temperatures to be transformed
    densities = torch.empty_like(temp)
    if use_gpu:
        temp = temp.cuda()
        densities = densities.cuda()
    # return densities
    # print(densities.size()
    # print(temp.size())
    densities[:] = 1000*(1-((temp[:]+288.9414)*torch.pow(temp[:] -
                         3.9863, 2))/(508929.2*(temp[:]+68.12963)))
    # densities[:] = 1000*(1-((temp[:]+288.9414)*torch.pow(temp[:] - 3.9863))/(508929.2*(temp[:]+68.12963)))
    # print("DENSITIES")
    # for i in range(10):
    #     print(densities[i,i])

    return densities

# Iterator through multiple dataloaders


class MyIter(object):
    """An iterator."""

    def __init__(self, my_loader):
        self.my_loader = my_loader
        self.loader_iters = [iter(loader) for loader in self.my_loader.loaders]
        # print("init", self.loader_iters)

    def __iter__(self):
        return self

    def __next__(self):
        # When the shortest loader (the one with minimum number of batches)
        # terminates, this iterator will terminates.
        # The `StopIteration` raised inside that shortest loader's `__next__`
        # method will in turn gets out of this `__next__` method.
        # print("next",     print(self.loader_iters))

        # raw
        # batches = [loader_iter.next() for loader_iter in self.loader_iters]

        batches = [next(loader_iter) for loader_iter in self.loader_iters]

        return self.my_loader.combine_batch(batches)

    # Python 2 compatibility
    next = __next__

    def __len__(self):
        return len(self.my_loader)

# wrapper class for multiple dataloaders


class MultiLoader(object):
    """This class wraps several pytorch DataLoader objects, allowing each time 
    taking a batch from each of them and then combining these several batches 
    into one. This class mimics the `for batch in loader:` interface of 
    pytorch `DataLoader`.
    Args: 
      loaders: a list or tuple of pytorch DataLoader objects
    """

    def __init__(self, loaders):
        self.loaders = loaders

    def __iter__(self):
        return MyIter(self)

    def __len__(self):
        l = min([len(loader) for loader in self.loaders])
        # print(l)
        return l

    # Customize the behavior of combining batches here.
    def combine_batch(self, batches):
        return batches


def xavier_normal_(tensor, gain=1.):
    # type: (Tensor, float) -> Tensor
    r"""Fills the input `Tensor` with values according to the method
    described in `Understanding the difficulty of training deep feedforward
    neural networks` - Glorot, X. & Bengio, Y. (2010), using a normal
    distribution. The resulting tensor will have values sampled from
    :math:`\mathcal{N}(0, \text{std})` where

    .. math::
        \text{std} = \text{gain} \times \sqrt{\frac{2}{\text{fan\_in} + \text{fan\_out}}}

    Also known as Glorot initialization.

    Args:
        tensor: an n-dimensional `torch.Tensor`
        gain: an optional scaling factor

    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.xavier_normal_(w)
    """
    fan_in, fan_out = _calculate_fan_in_and_fan_out(tensor)
    std = gain * math.sqrt(2.0 / float(fan_in + fan_out))

    return _no_grad_normal_(tensor, 0., std)