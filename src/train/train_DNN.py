import pandas as pd
import numpy as np
import sys
import os
import argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from data.lakePreprocess import DROP_COLUMNS, USE_FEATURES_COLUMNS, FLUX_COLUMNS, USE_FEATURES_COLUMNS_LAYER
from data.pytorch_data_operations import buildManyLakeDataByIds
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
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import os
from utils.train_utils import MyLSTM, set_seed, random_seed
from tqdm import tqdm
import pandas as pd
from sklearn.preprocessing import StandardScaler
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

def get_inputs_dnn(lake_id, cluster_id, COLUMNS_USE, seed):
    # COLUMNS_USE = USE_FEATURES_COLUMNS_LAYER
    load_path_main = f"../../models/41/seed={seed}/pril/group_{cluster_id}/fine_tune/individual_train_on_obs/"
    model_name = f"{lake_id}_pril_fine_tune_train_on_obs"
    load_path = load_path_main+ model_name
    data_dir =  f'../../data/processed/' # no-need

    ids = [lake_id]
    n_features = len(COLUMNS_USE) + 4
    use_gpu = False
    seq_length = 364
    win_shift = 364
    begin_loss_ind = 0


    layer_extended = 4 # Increasing the dimension representing the layer from 1 to 5.
    FLUX_START = n_features + layer_extended 

    (trn_data, _, val_data, val_dates, test_data, _)  = buildManyLakeDataByIds(ids, data_dir, seq_length, n_features, win_shift, use_obs = True, evaluate = True)

    batch_size = val_data.shape[0]
    if batch_size == 0:
        print("return none")
        return np.empty((0, 15)), np.empty((0,))
    n_hidden = torch.load(load_path)['state_dict']['out.weight'].shape[1]


    lstm_net = MyLSTM(n_features, n_hidden, batch_size, use_gpu)
    pretrain_dict = torch.load(load_path)['state_dict']
    model_dict = lstm_net.state_dict()
    pretrain_dict = {k: v for k, v in pretrain_dict.items() if k in model_dict}
    model_dict.update(pretrain_dict)
    lstm_net.load_state_dict(pretrain_dict)

    #things needed to predict test data
    mse_criterion = nn.MSELoss()
    mae_criterion = nn.L1Loss()
    testloader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    print("--------")
    lstm_net.eval()
    with torch.no_grad():
        avg_mse = 0
        avg_epi_mse = 0
        avg_hypo_mse = 0

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
            tmp_dates = val_dates[:, begin_loss_ind:]
            depths = inputs[:,:,0]

            #run model
            h_state = None
            lstm_net.hidden = lstm_net.init_hidden(batch_size=inputs.size()[0])
            pred, h_state = lstm_net(inputs, h_state)
            pred = pred.view(pred.size()[0],-1)
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



    assert ct == 1, f"Error count should be 1 not {ct}"
    # Mixed Pred, Target and Input
    zoom = 1.5
    mixed_pred = pred[::2,:]
    mixed_target = obs_targets[::2,:]
    mixed_se = (mixed_pred - mixed_target) ** 2
    mixed_se = mixed_se.cpu().numpy()
    mixed_se[np.isnan(mixed_se)] = 0
    mixed_se[upper_loss_indices[::2,:]] = 0
    mixed_se[mixed_se < mixed_mse.item()*zoom] = 0
    mixed_se[mixed_se > mixed_mse.item()*zoom] = 1

    # Upper layer Pred, Target and Input
    hypo_pred = pred[::2,:]
    hypo_target = obs_targets[::2,:]
    hypo_se = (hypo_pred - hypo_target) ** 2
    hypo_se = hypo_se.cpu().numpy()
    hypo_se[np.isnan(hypo_se)] = 0
    hypo_se[mixed_layer_indices[::2,:]] = 0
    hypo_se[hypo_se < upper_mse.item()*zoom] = 0
    hypo_se[hypo_se > upper_mse.item()*zoom] = 1

    # Lower layer Pred, Target and Input
    epi_pred = pred[1::2, :]
    epi_target = obs_targets[1::2,:]
    epi_se = (epi_pred - epi_target) ** 2
    epi_se = epi_se.cpu().numpy()  # Ensure tensor is on CPU before converting to NumPy
    epi_se[np.isnan(epi_se)] = 0
    lower_mse_value = lower_mse.item()  # Convert PyTorch tensor to a scalar
    epi_se[epi_se < lower_mse_value*zoom] = 0
    epi_se[epi_se > lower_mse_value*zoom] = 1

    label = np.logical_or.reduce((mixed_se, hypo_se, epi_se)).astype(int)
    num_ones = np.sum(label)
    num_zeros = label.size - num_ones



    loss_indices = loss_indices * true_upper_layer_mask
    label = label[loss_indices[::2,:]].flatten()
    num_ones = np.sum(label)

    inputs_dnn = inputs[::2, :, :15]
    inputs_dnn = inputs_dnn[loss_indices[::2, :]]  # 按照 loss_indices 进行过滤
    # 将过滤后的输入展平成二维矩阵
    inputs_dnn = inputs_dnn.reshape(-1, inputs_dnn.shape[-1])

    return inputs_dnn, label

class DNNModel(nn.Module):
    def __init__(self, input_size):
        super(DNNModel, self).__init__()
        self.fc1 = nn.Linear(input_size, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = self.sigmoid(self.fc4(x))
        return x

def train_dnn(inputs, label, cluster_id, seed):
    inputs_dnn = torch.tensor(inputs, dtype=torch.float32)
    labels = torch.tensor(label, dtype=torch.float32)

    # Reshape inputs and labels
    # inputs_dnn = inputs_dnn.view(-1, inputs_dnn.shape[2])  # Flatten to [batch_size*sequence_length, num_features]
    # labels = labels.view(-1, 1)  # Flatten to [batch_size*sequence_length, 1]


    # Define model, loss function, and optimizer
    input_size = inputs_dnn.shape[1]
    model = DNNModel(input_size)
    criterion = nn.BCELoss()  # Binary Cross Entropy Loss
    optimizer = optim.Adam(model.parameters(), lr=0.0005)

    # Split the dataset into training and testing sets
    dataset = TensorDataset(inputs_dnn, labels)
    train_size = int(0.9 * len(dataset))
    test_size = len(dataset) - train_size
    train_dataset, test_dataset = random_split(dataset, [train_size, test_size])

    # Create DataLoader for training and testing
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Training the model
    num_epochs = 500
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        for inputs_batch, labels_batch in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs_batch)
            labels_batch = labels_batch.unsqueeze(1)
            loss = criterion(outputs, labels_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {running_loss/len(train_loader)}')

    print('Training complete')

    # Save the model
    save_dir = f'../../models/41/seed={seed}/DNN'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'dnn_model_cluster_{cluster_id}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Model saved to {save_path}')

    # Evaluate the model on the test data
    model.eval()
    with torch.no_grad():
        total = 0
        correct = 0
        tp = 0  # True positives
        fp = 0  # False positives
        fn = 0  # False negatives

        for inputs_batch, labels_batch in test_loader:
            outputs = model(inputs_batch)
            predicted_labels = (outputs > 0.5).float()  # Convert probabilities to 0 or 1
            total += labels_batch.size(0)
            correct += (predicted_labels == labels_batch).sum().item()
            
            tp += ((predicted_labels == 1) & (labels_batch == 1)).sum().item()
            fp += ((predicted_labels == 1) & (labels_batch == 0)).sum().item()
            fn += ((predicted_labels == 0) & (labels_batch == 1)).sum().item()

        accuracy = correct / total
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        print(f'Overall Accuracy: {accuracy:.4f}')
        print(f'Precision: {precision:.4f}')
        print(f'Recall: {recall:.4f}')
        print(f'F1 Score: {f1_score:.4f}')

        print("number 1:", predicted_labels[predicted_labels == 1].sum())

def main(args):
    inputs_dnn = None
    labels = None
    model_index = args.model_index
    cluster_id = args.cluster_id
    IfTrainDnn = args.TrainDNN
    # for model_index in range(1,4):
    #     for cluster_id in range(1,5):
    ids = pd.read_csv(f'../../data/utils/groups/vol_area/new_clusters/cluster_{cluster_id}.csv')   
    ids = ids['nhdhr_id'].to_list()
    COLUMNS_USE = USE_FEATURES_COLUMNS_LAYER
    seed = random_seed[model_index - 1]
    set_seed(seed)
    for i, lake_id in enumerate(ids):
        print("Processing lake:", lake_id)
        inputs, label = get_inputs_dnn(lake_id, cluster_id, COLUMNS_USE, seed)
        if inputs_dnn is None or labels is None:
            inputs_dnn = inputs
            labels = label
        else:
            inputs_dnn = np.concatenate((inputs_dnn, inputs), axis=0)
            labels = np.concatenate((labels, label), axis=0)
    if IfTrainDnn:
        train_dnn(inputs_dnn, labels, cluster_id, seed)

    ids = pd.read_csv(f'../../data/utils/groups/vol_area/cluster_{cluster_id}.csv') # 425 lakes
    base_path = '../../data'
    read_path = base_path + '/norm/'
    read_raw_path = base_path + '/raw/'
    model_path = f'../../models/41/seed={seed}/DNN/dnn_model_cluster_{cluster_id}.pth'
    usecols = ['sat_hypo', 'thermocline_depth',
        'temperature_epi', 'temperature_hypo', 'volume_epi', 'volume_hypo',
        'wind', 'airtemp', 'fnep', 'fmineral', 'fsed', 'fatm', 'fdiff',
        'fentr_epi', 'fentr_hyp']

    # usecols = ['thermocline_depth', 'volume_epi', 'volume_hypo','fnep', 'fmineral', 'fsed', 'fatm', 'fdiff',
    #        'fentr_epi', 'fentr_hyp']

    # usecols = ['fentr_epi', 'fentr_hyp', 'sim_epi', 'sim_hyp']

    # sim_cols = ['sim_epi', 'sim_hyp']
    # usecols = ['fnep', 'fmineral', 'fsed', 'fatm', 'fdiff', 'fentr_epi', 'fentr_hyp']

    # , 'sim_epi', 'sim_hyp'
    # usecols = ['sat_hypo', 'thermocline_depth',
    #        'temperature_epi', 'temperature_hypo', 'volume_epi', 'volume_hypo',
    #        'wind', 'airtemp', 'fnep', 'fmineral', 'fsed', 'fatm', 'fdiff',
    #        'fentr_epi', 'fentr_hyp', 'eutro', 'oligo', 'dys', 'water', 'developed',
    #        'barren', 'forest', 'shrubland', 'herbaceous', 'cultivated', 'wetlands',
    #        'depth', 'area', 'elev', 'Shore_len', 'Vol_total', 'Vol_res', 'Vol_src',
    #        'Depth_avg', 'Dis_avg', 'Res_time', 'Elevation', 'Slope_100',
    #        'Wshd_area', 'sim_epi', 'sim_hyp']

    def save_extend_file(save_path, data_pd):
        # Save the merged DataFrame
        data_pd.to_csv(save_path, index=False)
        print(f'Saved extended data for lake {nid} to {save_path}')

    FLUX_COLUMNS = ['fnep', 'fmineral', 'fsed', 'fatm', 'fdiff', 'fentr_epi', 'fentr_hyp']
    Inter_columns = ['sat_hypo', 'thermocline_depth','temperature_epi', 'temperature_hypo', 'volume_epi', 'volume_hypo']


    for number, lake_id in tqdm(enumerate(ids['nhdhr_id'])):
        nid = str(lake_id)
        norm_data_df = pd.read_csv(read_path + nid + '.csv')
        raw_data_df = pd.read_csv(read_raw_path + nid + '.csv')
        inputs_df = pd.read_csv(read_path + nid + '.csv', usecols=usecols)

        flux_df = pd.read_csv(read_path + nid + '.csv', usecols=['fnep', 'fmineral', 'fsed', 'fatm', 'fdiff', 'fentr_epi', 'fentr_hyp'])
        flux_df = flux_df.to_numpy()
        # sim_data_df = pd.read_csv(read_path + nid + '.csv', usecols=sim_cols)
        # sim_data = sim_data_df.to_numpy()

        # Standardize the inputs
        # scaler = StandardScaler()
        # inputs = scaler.fit_transform(inputs_df)
        inputs = inputs_df.to_numpy()
        # inputs = np.concatenate([inputs, sim_data], axis=1)

        # Convert inputs to torch tensor
        inputs_tensor = torch.tensor(inputs, dtype=torch.float32)

        input_size = inputs_tensor.shape[1]
        model = DNNModel(input_size)
        model.load_state_dict(torch.load(model_path))
        # Make predictions with the DNN model
        with torch.no_grad():
            predictions = model(inputs_tensor).numpy()
        
        binary_predictions = (predictions >= 0.5).astype(int)
        
        # Setting Rules
        rule = np.zeros_like(binary_predictions)
        sim_hypo = norm_data_df['sim_hyp'].values
        sim_epi = norm_data_df['sim_hyp'].values
        sim_hypo_pre = np.roll(sim_hypo, 1)  # Shift sim_hypo by one to get previous day's values
        sim_epi_pre = np.roll(sim_epi, 1)
        # Calculate relative change
        epsilon = 1e-10
        ratio = 0.2 # (Hyperparameter)
        sim_hypo_pre_safe = np.where(sim_hypo_pre == 0, epsilon, sim_hypo_pre)
        sim_epi_pre_safe = np.where(sim_epi_pre == 0, epsilon, sim_epi_pre)

        # Calculate relative change
        relative_change_hypo = np.abs((sim_hypo[1:] - sim_hypo_pre[1:]) / sim_hypo_pre_safe[1:])
        relative_change_epi = np.abs((sim_epi[1:] - sim_epi_pre[1:]) / sim_epi_pre_safe[1:])

        extend_based_on_change_hypo = np.zeros_like(sim_hypo, dtype=bool)
        extend_based_on_change_hypo[1:] = relative_change_hypo > ratio

        extend_based_on_change_epi = np.zeros_like(sim_epi, dtype=bool)
        extend_based_on_change_epi[1:] = relative_change_epi > ratio
        extend_based_on_change = np.logical_or(extend_based_on_change_hypo, extend_based_on_change_epi)

        extend_based_on_change = np.expand_dims(extend_based_on_change, axis=1)

        print("count 1 in extend_based_on_change:", np.sum(extend_based_on_change))
        print("count 1 in binary_predictions:", np.sum(binary_predictions))
        
        final_if_extend = np.zeros_like(binary_predictions)
        print("extend_based_on_change shape:", extend_based_on_change.shape)
        final_if_extend = binary_predictions * extend_based_on_change
        print("final_if_extend shape:", final_if_extend.shape)
        # Add predictions to the raw data DataFrame
        # raw_data_df['if_extend'] = binary_predictions
        norm_data_df.loc[norm_data_df['mixed'] == 0, 'extend'] = final_if_extend[norm_data_df['mixed'] == 0]
        raw_data_df.loc[norm_data_df['mixed'] == 0, 'extend'] = final_if_extend[norm_data_df['mixed'] == 0]
        print("sum of 1:", np.sum(norm_data_df['extend'] == 1))

        # Define the save path
        save_norm_path = os.path.join(base_path, f'extend_norm/seed={seed}/', nid + '.csv')
        os.makedirs(os.path.dirname(save_norm_path), exist_ok=True)

        save_raw_path = os.path.join(base_path, f'extend_raw/seed={seed}/', nid + '.csv')
        os.makedirs(os.path.dirname(save_raw_path), exist_ok=True)

        save_extend_file(save_raw_path, raw_data_df)
        save_extend_file(save_norm_path, norm_data_df)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model type.")
    # parser.add_argument("--model_type", type=str, default='lstm', help="get model type from input args",
    #                     choices=['lstm', 'pril', 'april', 'ea_lstm', 'transformer'])
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument("--model_index", type=int,
                        default=1, help="model_index[1,2,3] ===> ramdom seed[40,42,44]", choices=[1,2,3])
    parser.add_argument("--cluster_id", type=int,
                        default=1, help="Cluster Id", choices=[1,2,3,4])
    parser.add_argument("--TrainDNN", type=int,
                        default=1, help="Set to 1 to retrain the DNN model and generate data; set to 0 to load an existing DNN model and generate data.", choices=[0, 1])
    args = parser.parse_args()
    main(args)
