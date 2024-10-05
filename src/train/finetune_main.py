
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
from utils.train_utils import MyLSTM, set_seed, random_seed#
def main(args):
    model_index = args.model_index
    cluster_id = args.cluster_id
    model_type = args.model_type
    seed = random_seed[model_index-1]
    set_seed(seed)
    ids = pd.read_csv(
        f'../../data/utils/groups/vol_area/new_clusters/cluster_{cluster_id}.csv')
    ids = ids['nhdhr_id'].to_list()
    # ids = ['nhdhr_120018092']
    for count, lake_id in enumerate(ids):
        print("lake id:", lake_id)
        if model_type == 'april':
            from finetune import train_finetune_April as finetune
        elif model_type == 'pril':
            from finetune import train_finetune_Pril as finetune
        elif model_type == 'lstm':
            from finetune import train_finetune_Lstm as finetune
        elif model_type == 'ea_lstm':
            from finetune import train_finetune_EaLstm as finetune 
        elif model_type == 'transformer':
            from finetune import train_finetune_Transformer as finetune

        finetune(lake_id, cluster_id, seed, model_index)
        


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
