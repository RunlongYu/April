import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
import random
import math
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.clip_grad import clip_grad_norm_
import torch.optim as optim
import json


from models.pytorch_model_operations import saveModel
from data.pytorch_data_operations import buildOneLakeData, buildManyLakeDataByIds, calculate_total_DOC_conservation_loss, calculate_stratified_DOC_conservation_loss
from data.lakePreprocess import USE_FEATURES_COLUMNS, USE_FEATURES_COLUMNS_LAYER, USE_FEATURES_COLUMNS_NOFLUX, FLUX_COLUMNS, FLUX_START
from typing import Dict, List, Tuple


class MyEALSTM(nn.Module):
    """Wrapper class that connects LSTM/EA-LSTM with fully connected layer"""

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 initial_forget_bias: int = 5,
                 dropout: float = 0.0,
                 concat_static: bool = False,
                 no_static: bool = False):
        """Initialize model.
        Parameters
        ----------
        input_size_dyn: int
            Number of dynamic input features.
        input_size_stat: int
            Number of static input features (used in the EA-LSTM input gate).
        hidden_size: int
            Number of LSTM cells/hidden units.
        initial_forget_bias: int
            Value of the initial forget gate bias. (default: 5)
        dropout: float
            Dropout probability in range(0,1). (default: 0.0)
        concat_static: bool
            If True, uses standard LSTM otherwise uses EA-LSTM
        no_static: bool
            If True, runs standard LSTM
        """
        super(MyEALSTM, self).__init__()
        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.initial_forget_bias = initial_forget_bias
        self.dropout_rate = dropout
        self.concat_static = concat_static
        self.no_static = no_static
        self.lstm = EALSTM(input_size_dyn=input_size_dyn,
                           input_size_stat=input_size_stat,
                           hidden_size=hidden_size,
                           initial_forget_bias=initial_forget_bias)

        self.dropout = nn.Dropout(p=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor = None) \
            -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run forward pass through the model.

        Parameters
        ----------
        x_d : torch.Tensor
            Tensor containing the dynamic input features of shape [batch, seq_length, n_features]
        x_s : torch.Tensor, optional
            Tensor containing the static catchment characteristics, by default None

        Returns
        -------
        out : torch.Tensor
            Tensor containing the network predictions of shape [batch, seq_length, 1]
        h_n : torch.Tensor
            Tensor containing the hidden states of each time step
        c_n : torch.Tensor
            Tensor containing the cell states of each time step
        """
        if self.concat_static or self.no_static:
            h_n, c_n = self.lstm(x_d)
        else:
            h_n, c_n = self.lstm(x_d, x_s)

        h_n = self.dropout(h_n)

        # Apply fully connected layer to each time step
        out = self.fc(h_n)

        return out, h_n, c_n


class EALSTM(nn.Module):
    """Implementation of the Entity-Aware-LSTM (EA-LSTM)

    TODO: Include paper ref and latex equations

    Parameters
    ----------
    input_size_dyn : int
        Number of dynamic features, which are those, passed to the LSTM at each time step.
    input_size_stat : int
        Number of static features, which are those that are used to modulate the input gate.
    hidden_size : int
        Number of hidden/memory cells.
    batch_first : bool, optional
        If True, expects the batch inputs to be of shape [batch, seq, features] otherwise, the
        shape has to be [seq, batch, features], by default True.
    initial_forget_bias : int, optional
        Value of the initial forget gate bias, by default 0

    """

    def __init__(self,
                 input_size_dyn: int,
                 input_size_stat: int,
                 hidden_size: int,
                 batch_first: bool = True,
                 initial_forget_bias: int = 0):
        super(EALSTM, self).__init__()

        self.input_size_dyn = input_size_dyn
        self.input_size_stat = input_size_stat
        self.hidden_size = hidden_size
        self.batch_first = batch_first
        self.initial_forget_bias = initial_forget_bias

        # create tensors of learnable parameters
        self.weight_ih = nn.Parameter(
            torch.FloatTensor(input_size_dyn, 3 * hidden_size))
        self.weight_hh = nn.Parameter(
            torch.FloatTensor(hidden_size, 3 * hidden_size))
        self.weight_sh = nn.Parameter(
            torch.FloatTensor(input_size_stat, hidden_size))
        self.bias = nn.Parameter(torch.FloatTensor(3 * hidden_size))
        self.bias_s = nn.Parameter(torch.FloatTensor(hidden_size))

        # initialize parameters
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize all learnable parameters of the LSTM"""
        nn.init.orthogonal_(self.weight_ih.data)
        nn.init.orthogonal_(self.weight_sh)

        weight_hh_data = torch.eye(self.hidden_size)
        weight_hh_data = weight_hh_data.repeat(1, 3)
        self.weight_hh.data = weight_hh_data

        nn.init.constant_(self.bias.data, val=0)
        nn.init.constant_(self.bias_s.data, val=0)

        if self.initial_forget_bias != 0:
            self.bias.data[:self.hidden_size] = self.initial_forget_bias

    def forward(self, x_d: torch.Tensor, x_s: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """[summary]

        Parameters
        ----------
        x_d : torch.Tensor
            Tensor, containing a batch of sequences of the dynamic features. Shape has to match
            the format specified with batch_first.
        x_s : torch.Tensor
            Tensor, containing a batch of static features.

        Returns
        -------
        h_n : torch.Tensor
            The hidden states of each time step of each sample in the batch.
        c_n : torch.Tensor]
            The cell states of each time step of each sample in the batch.
        """
        if self.batch_first:
            x_d = x_d.transpose(0, 1)

        seq_len, batch_size, _ = x_d.size()

        h_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        c_0 = x_d.data.new(batch_size, self.hidden_size).zero_()
        h_x = (h_0, c_0)

        # empty lists to temporally store all intermediate hidden/cell states
        h_n, c_n = [], []

        # expand bias vectors to batch size
        bias_batch = (self.bias.unsqueeze(0).expand(
            batch_size, *self.bias.size()))

        # calculate input gate only once because inputs are static
        bias_s_batch = (self.bias_s.unsqueeze(
            0).expand(batch_size, *self.bias_s.size()))
        i = torch.sigmoid(torch.addmm(bias_s_batch, x_s, self.weight_sh))

        # perform forward steps over input sequence
        for t in range(seq_len):
            h_0, c_0 = h_x

            # calculate gates
            gates = (torch.addmm(bias_batch, h_0, self.weight_hh) +
                     torch.mm(x_d[t], self.weight_ih))
            f, o, g = gates.chunk(3, 1)

            c_1 = torch.sigmoid(f) * c_0 + i * torch.tanh(g)
            h_1 = torch.sigmoid(o) * torch.tanh(c_1)

            # store intermediate hidden/cell state in list
            h_n.append(h_1)
            c_n.append(c_1)

            h_x = (h_1, c_1)

        h_n = torch.stack(h_n, 0)
        c_n = torch.stack(c_n, 0)

        if self.batch_first:
            h_n = h_n.transpose(0, 1)
            c_n = c_n.transpose(0, 1)

        return h_n, c_n


class TransformerModel(nn.Module):
    def __init__(self, input_size, num_heads, num_layers, hidden_size, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.input_size = input_size
        self.model_type = 'Transformer'
        self.encoder_layer = nn.TransformerEncoderLayer(
            d_model=input_size, nhead=num_heads, dim_feedforward=hidden_size, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(
            self.encoder_layer, num_layers=num_layers)
        self.decoder = nn.Linear(input_size, 1)  # Assuming 1 output feature

    def forward(self, src):
        # Transformer expects input of shape (S, N, E)
        src = src.permute(1, 0, 2)
        output = self.transformer_encoder(src)
        output = self.decoder(output)
        output = output.permute(1, 0, 2)  # Back to (N, S, E)
        return output


def _process_one_batch(args, device, model, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
    batch_x = batch_x.float().to(device)
    batch_y = batch_y.float()

    batch_x_mark = batch_x_mark.float().to(device)
    batch_y_mark = batch_y_mark.float().to(device)

    # decoder input
    if args.padding == 0:
        dec_inp = torch.zeros(
            [batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
    elif args.padding == 1:
        dec_inp = torch.ones(
            [batch_y.shape[0], args.pred_len, batch_y.shape[-1]]).float()
    dec_inp = torch.cat([batch_y[:, :args.label_len, :],
                        dec_inp], dim=1).float().to(device)
    # encoder - decoder

    outputs = model(batch_x, batch_x_mark, dec_inp, batch_y_mark)

    return outputs, batch_y


class MyLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size, use_gpu):
        super(MyLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.use_gpu = use_gpu
        self.lstm = nn.LSTM(input_size=input_size,
                            hidden_size=hidden_size, batch_first=True)
        self.out = nn.Linear(hidden_size, 1)
        self.hidden = self.init_hidden()

    def init_hidden(self, batch_size=0):
        # initialize both hidden layers
        if batch_size == 0:
            batch_size = self.batch_size
        ret = (xavier_normal_(torch.empty(1, batch_size, self.hidden_size)),
               xavier_normal_(torch.empty(1, batch_size, self.hidden_size)))
        if self.use_gpu:
            item0 = ret[0].cuda(non_blocking=True)
            item1 = ret[1].cuda(non_blocking=True)
            ret = (item0, item1)
        return ret

    def forward(self, x, hidden):
        self.lstm.flatten_parameters()
        x = x.float()
        x, hidden = self.lstm(x, self.hidden)
        self.hidden = hidden
        x = self.out(x)
        x = F.relu(x, inplace=False)  # 确保inplace=False
        return x, hidden


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


random_seed = [40, 42, 44]


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


class TransformerConfig:
    def __init__(self, learning_rate=0.002, batch_size=32, num_heads=5, num_layers=2, hidden_size=32):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.hidden_size = hidden_size


class PretrainLSTMCfg:
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.train_epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.lambda_total = None
        self.lambda_stratified_epi = None
        self.lambda_stratified_hypo = None
        self.IS_PGRNN = False
        self.USE_EXTEND = False
        self.set_params(cluster_id)

    def set_params(self, cluster_id):
        if cluster_id == 1:
            self.train_epochs = 8
            self.batch_size = 32
            self.learning_rate = .005
            self.lambda_total = 0
            self.lambda_stratified_epi = 0.0
            self.lambda_stratified_hypo = 0.0
        elif cluster_id == 2:
            self.train_epochs = 8
            self.batch_size = 32
            self.learning_rate = .005
            self.lambda_total = 0
            self.lambda_stratified_epi = 0.0
            self.lambda_stratified_hypo = 0.0
        elif cluster_id == 3:
            self.train_epochs = 8
            self.batch_size = 32
            self.learning_rate = .005
            self.lambda_total = 0
            self.lambda_stratified_epi = 0.0
            self.lambda_stratified_hypo = 0.0
        elif cluster_id == 4:
            self.train_epochs = 8
            self.batch_size = 32
            self.learning_rate = .005
            self.lambda_total = 0
            self.lambda_stratified_epi = 0.0
            self.lambda_stratified_hypo = 0.0
        else:
            raise ValueError("Invalid cluster ID. Must be 1, 2, 3, or 4.")


class PretrainPrilCfg:
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.train_epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.lambda_total = None
        self.lambda_stratified_epi = None
        self.lambda_stratified_hypo = None
        self.IS_PGRNN = True
        self.USE_EXTEND = False
        self.set_params(cluster_id)

    def set_params(self, cluster_id):
        if cluster_id == 1:
            self.train_epochs = 8
            self.batch_size = 32
            self.learning_rate = .0025
            self.lambda_total = 1
            self.lambda_stratified_epi = 2
            self.lambda_stratified_hypo = 3
        elif cluster_id == 2:
            self.train_epochs = 8
            self.batch_size = 32
            self.learning_rate = .0025
            self.lambda_total = 1
            self.lambda_stratified_epi = 2
            self.lambda_stratified_hypo = 3
        elif cluster_id == 3:
            self.train_epochs = 8
            self.batch_size = 32
            self.learning_rate = .0025
            self.lambda_total = 1
            self.lambda_stratified_epi = 2
            self.lambda_stratified_hypo = 3
        elif cluster_id == 4:
            self.train_epochs = 8
            self.batch_size = 32
            self.learning_rate = .0025
            self.lambda_total = 1
            self.lambda_stratified_epi = 2
            self.lambda_stratified_hypo = 3
        else:
            raise ValueError("Invalid cluster ID. Must be 1, 2, 3, or 4.")


class PretrainAprilCfg:
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.train_epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.lambda_total = None
        self.lambda_stratified_epi = None
        self.lambda_stratified_hypo = None
        self.IS_PGRNN = True
        self.USE_EXTEND = True
        self.set_params(cluster_id)

    def set_params(self, cluster_id):
        if cluster_id == 1:
            self.train_epochs = 20
            self.batch_size = 32
            self.learning_rate = .0025
            self.lambda_total = 1
            self.lambda_stratified_epi = 4
            self.lambda_stratified_hypo = 6.5
        ################################
        elif cluster_id == 2:
            self.train_epochs = 5
            self.batch_size = 16
            self.learning_rate = .005
            self.lambda_total = 1
            self.lambda_stratified_epi = 4
            self.lambda_stratified_hypo = 6
        elif cluster_id == 3:
            self.train_epochs = 8
            self.batch_size = 32
            self.learning_rate = .0108
            self.lambda_total = 1
            self.lambda_stratified_epi = 3
            self.lambda_stratified_hypo = 5
        elif cluster_id == 4:
            self.train_epochs = 8
            self.batch_size = 32
            self.learning_rate = .0095
            self.lambda_total = 1
            self.lambda_stratified_epi = 3
            self.lambda_stratified_hypo = 5
        else:
            raise ValueError("Invalid cluster ID. Must be 1, 2, 3, or 4.")


class FineTuneAprilCfg:
    def __init__(self, cluster_id):
        self.cluster_id = cluster_id
        self.train_epochs = None
        self.batch_size = None
        self.learning_rate = None
        self.lambda_total = None
        self.lambda_stratified_epi = None
        self.lambda_stratified_hypo = None
        self.set_params(cluster_id)

    def set_params(self, cluster_id):
        if cluster_id == 1:
            self.train_epochs = 12
            self.batch_size = 8
            self.learning_rate = .005
            self.lambda_total = 1
            self.lambda_stratified_epi = 3.5
            self.lambda_stratified_hypo = 5.3
        elif cluster_id == 2:
            self.train_epochs = 11
            self.batch_size = 8
            self.learning_rate = .0045
            self.lambda_total = 1
            self.lambda_stratified_epi = 3
            self.lambda_stratified_hypo = 3
        elif cluster_id == 3:
            self.train_epochs = 10
            self.batch_size = 8
            self.learning_rate = .004
            self.lambda_total = 1
            self.lambda_stratified_epi = 2
            self.lambda_stratified_hypo = 5
        elif cluster_id == 4:
            self.train_epochs = 11
            self.batch_size = 8
            self.learning_rate = .004
            self.lambda_total = 1
            self.lambda_stratified_epi = 3
            self.lambda_stratified_hypo = 6
        else:
            raise ValueError("Invalid cluster ID. Must be 1, 2, 3, or 4.")
