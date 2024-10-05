import argparse
import sys
import os
from time import sleep
import pandas as pd
import time
import torch
########### MY PACKAGES ###########
# sys.path.append(os.path.dirname(sys.path[0]))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


def main(args):
    model_type = args.model_type
    if model_type == 'lstm' or model_type == 'pril' or model_type == 'april':
        from pretrain import pretrain_model
    if model_type == 'ea_lstm':
        from pretrain import pretrain_ealstm as pretrain_model
    if model_type == 'transformer':
        from pretrain import pretrain_transformer as pretrain_model
    pretrain_model(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Process model type.")
    parser.add_argument("--model_type", type=str, default='lstm', help="get model type from input args",
                        choices=['lstm', 'pril', 'april', 'ea_lstm', 'transformer'])
    parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
    parser.add_argument("--model_index", type=int,
                        default=1, help="model_index[1,2,3] ===> ramdom seed[40,42,44]")
    parser.add_argument("--cluster_id", type=int,
                        default=1, help="Cluster Id")
    args = parser.parse_args()
    main(args)
