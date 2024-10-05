from lakePreprocess import USE_FEATURES_COLUMNS, FLUX_COLUMNS
import pandas as pd
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.train_utils import MyLSTM, set_seed, random_seed


def save_npy_data(COLUMNS_USE, seed):
    debug = False
    my_path = os.path.abspath(os.path.dirname(__file__))
    base_path = my_path+ '/../../data'
    ids = pd.read_csv('../../data/utils/ids.csv') # 425 lakes
    n_features = len(COLUMNS_USE) + 1 # +1 represent "mixed"

    n_flux = len(FLUX_COLUMNS)
    raw_data_path = base_path + f'/extend_raw/seed={seed}/'
    read_path = base_path + f'/extend_norm/seed={seed}/'
    # read_path = base_path + '/extend/'

    for number, lake_id in enumerate(ids['nhdhr_id']):
        nid = lake_id
        name = str(nid)
        nid = str(nid)
        if debug:
            print(number," starting laka id:", name)
        #read/format dates
        dates_df = pd.read_csv(read_path+nid+'.csv', usecols=['datetime'])
        dates_df['datetime'] = pd.to_datetime(dates_df['datetime'])
        dates = dates_df['datetime'].values

        if debug:
            print("dates shape",dates.shape)
            print(dates[0])

        # 2 layers : fixed in Oxygen prediction project
        n_layer = 2 
        # Load obs
        obs = pd.read_csv(read_path+nid+'.csv', usecols= ['obs_epi','obs_hyp']) 
        obs_values = obs.to_numpy()
        # Load sim data
        sim = pd.read_csv(read_path+nid+'.csv', usecols=['sim_epi', 'sim_hyp']) 
        sim_values = sim.to_numpy()
        sim_pre_train = sim.copy()


        if debug:
            print("sim shape:", sim_values.shape)
            print("sim nan:",np.count_nonzero(np.isnan(sim_values)))
            print("sim not nan", np.count_nonzero(~np.isnan(sim_values)))
            print("obs shape:", obs_values.shape)
            print("obs nan:",np.count_nonzero(np.isnan(obs_values)))
            print("obs not nan", np.count_nonzero(~np.isnan(obs_values)))
            print("-----------------------------------------")

        start_date = dates[0]
        end_date = dates[-1]
        if debug:
            print(f"{lake_id} dates: from {start_date} -------> {end_date}")
            print("dates shape:", dates.shape)

        ####################### START LOADING FEATURES #########################
        features_df = pd.read_csv(read_path+nid+'.csv', usecols=COLUMNS_USE) 
        features_df = features_df.apply(pd.to_numeric, errors='coerce')
        features = features_df.to_numpy()

        mixed_df = pd.read_csv(read_path+nid+'.csv', usecols=['mixed']) 
        mixed = mixed_df.to_numpy()
        sim_pre_train = sim_pre_train.to_numpy()
        sim_pre_train[mixed[:, 0] == 1, 1] = np.nan
        
        if debug:
            print("feature nan:", features_df.isna().sum())

        if np.count_nonzero(np.isnan(features)) != 0:
            raise Exception("data in features missing")

    
        flux_df = pd.read_csv(raw_data_path+nid+'.csv', usecols=FLUX_COLUMNS)

        # pre processing volume and flux data
        # Volume
        # first_valid_index = flux_df['volume_epi'].first_valid_index()
    
        # if first_valid_index is not None:
        #     # 获取 'volume_epi' 和 'volume_hypo' 在此索引的值
        #     V_epi = flux_df.loc[first_valid_index, 'volume_epi']
        #     V_hypo = flux_df.loc[first_valid_index, 'volume_hypo']

        #     # 检查 'volume_hypo' 是否也是非 NaN，如果是 NaN，可以选择如何处理
        #     if pd.notna(V_hypo):
        #         V_total = V_epi + V_hypo
        #         # print(f"V_total at index {first_valid_index} is {V_total}")
        #     else:
        #         print(f"'volume_hypo' at index {first_valid_index} is NaN")
        # else:
        #     print("No valid 'volume_epi' value found in the DataFrame")

        # flux_df['volume_epi'].fillna(V_total, inplace=True)
        # flux_df['volume_hypo'].fillna(0, inplace=True)

        # Flux
        # flux_list1 = ['fnep', 'fmineral', 'fsed', 'fatm', 'fdiff']
        # flux_df.loc[0, flux_list1] = flux_df.loc[0, flux_list1].fillna(flux_df.loc[1, flux_list1])

        # flux_df['fentr_epi'].fillna(0, inplace=True)
        # flux_df['fentr_hyp'].fillna(0, inplace=True)
        # flux_df = flux_df.apply(pd.to_numeric, errors='coerce')
        flux = flux_df.to_numpy()

        assert not np.isnan(flux).any(), "Flux data contains NaN values after preprocessing"

        if debug:
            print("flux shape:", flux.shape)
        if debug:
            print("features nan:",np.count_nonzero(np.isnan(features)))
            print("features not nan", np.count_nonzero(~np.isnan(features)))
            
        
        # normalization
        columns_to_normalize = [i for i in range(features.shape[1]) if i < 8 or i >= 15]
        columns_to_exclude = [i for i in range(8, 15)]

        # # 选择需要标准化的列
        # features_to_normalize = features[:, columns_to_normalize]

        # 标准化
        # scaler = StandardScaler()
        # features_processed = scaler.fit_transform(features)
        features_processed = features
        # 重新组合数据
        # features_processed = np.empty_like(features)
        # features_processed[:, columns_to_normalize] = features_normalized
        # features_processed[:, columns_to_exclude] = features[:, columns_to_exclude]
        # flux_for_debug = pd.read_csv(read_path+nid+'.csv', usecols=['fnep', 'fmineral', 'fsed', 'fatm', 'fdiff', 'fentr_epi', 'fentr_hyp'])
        # flux_for_debug = flux_for_debug.to_numpy()

        # features_processed[:,8:15] = flux_for_debug
        # print("flux_for_debug:", flux_for_debug.shape)
        # print("features_processed[:, columns_to_exclude]:", features_processed[:, columns_to_exclude].shape)
        # print((features_processed[:, 8:15] == flux_for_debug).all())
        # assert (features_processed[:, 8:15] == flux_for_debug).all() == True, "not match flux"

        # add mixed feature     
        if debug:
            print("features_processed_no_mixed shape:", features_processed.shape)
        
        features_processed = np.column_stack((features_processed, mixed))
    

        if debug:
            print("mix_df",mixed_df)
            print("features_processed shape:", features_processed.shape)
        if debug:
            print("features_norm nan:",np.count_nonzero(np.isnan(features_processed)))
            print("features_norm not nan", np.count_nonzero(~np.isnan(features_processed)))
            print("features size", features_processed.shape)
            print("-----------------------------------------")

        n_dates = dates.shape[0]
        if n_dates != features_processed.shape[0]:
            raise Exception("dates dont match")
        ####################### FINISH LOADING FEATURES #########################

        ####################### START DATA(TRAIN/VAL/TEST) #########################
        first_train_date = start_date
        last_train_date = np.datetime64('2011-12-31')

        first_val_date = np.datetime64('2012-01-01')
        last_val_date = np.datetime64('2015-12-31')

        first_tst_date = np.datetime64('2016-01-01')
        last_tst_date = end_date

        train_indices = (dates >= first_train_date) & (dates <= last_train_date)
        val_indices = (dates >= first_val_date) & (dates <= last_val_date)
        test_indices = (dates >= first_tst_date) & (dates <= last_tst_date)

        n_dates_train = np.sum(train_indices)
        n_dates_val = np.sum(val_indices)
        n_dates_test = np.sum(test_indices)

        if debug:
            print("n_dates_train:", n_dates_train)
            print("n_dates_val:", n_dates_val)
            print("n_dates_test:", n_dates_test)
            print("-----------------------------------------")
        
        layer_dim = 5 # To help the model better learn the distinction between upper and lower layers, we added layer information with 5 dimensions.
        label_size = 1

        validation_label_size = 2

        trn_pre_train_data = np.empty((n_layer, n_dates_train, n_features + layer_dim + n_flux+ label_size))
        val_pre_train_data = np.empty((n_layer, n_dates_val, n_features + layer_dim + n_flux+ label_size))
        tst_pre_train_data = np.empty((n_layer, n_dates_test, n_features + layer_dim + n_flux+ label_size))
        trn_fine_tune_data = np.empty((n_layer, n_dates_train, n_features + layer_dim + n_flux+ label_size))
        val_fine_tune_data = np.empty((n_layer, n_dates_val, n_features + layer_dim + n_flux+ label_size))
        tst_fine_tune_data = np.empty((n_layer, n_dates_test, n_features + layer_dim + n_flux+ label_size))

        trn_evaluate_data = np.empty((n_layer, n_dates_train, n_features + layer_dim + n_flux+ validation_label_size))
        val_evaluate_data = np.empty((n_layer, n_dates_val, n_features + layer_dim + n_flux+ validation_label_size))
        tst_evaluate_data = np.empty((n_layer, n_dates_test, n_features + layer_dim + n_flux+ validation_label_size))

        trn_pre_train_data[:] = np.nan
        val_pre_train_data[:] = np.nan
        tst_pre_train_data[:] = np.nan
        trn_fine_tune_data[:] = np.nan
        val_fine_tune_data[:] = np.nan
        tst_fine_tune_data[:] = np.nan
        trn_evaluate_data[:] = np.nan
        val_evaluate_data[:] = np.nan
        tst_evaluate_data[:] = np.nan

        trn_dates = dates[train_indices]
        val_dates = dates[val_indices]
        tst_dates = dates[test_indices]
        if debug:
            print("trn_data shape :", trn_pre_train_data.shape)
            print("val_data shape:", trn_pre_train_data.shape)
            print("tst_data shape:", trn_pre_train_data.shape)
            print("-----------------------------------------")
    
        flux_data_start = n_features+layer_dim

        for l in range(n_layer):
            # pre-train data
            trn_pre_train_data[l,:,:n_features] = features_processed[train_indices]
            trn_pre_train_data[l,:,n_features:n_features+layer_dim] = l

            val_pre_train_data[l,:,:n_features] = features_processed[val_indices]
            val_pre_train_data[l,:,n_features:n_features+layer_dim] = l

            tst_pre_train_data[l,:,:n_features] = features_processed[test_indices]
            tst_pre_train_data[l,:,n_features:n_features+layer_dim] = l

            # fine-tune data
            trn_fine_tune_data[l,:,:n_features] = features_processed[train_indices]
            trn_fine_tune_data[l,:,n_features:n_features+layer_dim] = l

            val_fine_tune_data[l,:,:n_features] = features_processed[val_indices]
            val_fine_tune_data[l,:,n_features:n_features+layer_dim] = l

            tst_fine_tune_data[l,:,:n_features] = features_processed[test_indices]
            tst_fine_tune_data[l,:,n_features:n_features+layer_dim] = l

            # evaluate data
            trn_evaluate_data[l,:,:n_features] = features_processed[train_indices]
            trn_evaluate_data[l,:,n_features:n_features+layer_dim] = l

            val_evaluate_data[l,:,:n_features] = features_processed[val_indices]
            val_evaluate_data[l,:,n_features:n_features+layer_dim] = l

            tst_evaluate_data[l,:,:n_features] = features_processed[test_indices]
            tst_evaluate_data[l,:,n_features:n_features+layer_dim] = l
            
            ####### FLUX DATA #######
            trn_pre_train_data[l,:,flux_data_start:-label_size] = flux[train_indices]
            val_pre_train_data[l,:,flux_data_start:-label_size] = flux[val_indices]
            tst_pre_train_data[l,:,flux_data_start:-label_size] = flux[test_indices]

            trn_fine_tune_data[l,:,flux_data_start:-label_size] = flux[train_indices]
            val_fine_tune_data[l,:,flux_data_start:-label_size] = flux[val_indices]
            tst_fine_tune_data[l,:,flux_data_start:-label_size] = flux[test_indices]

            trn_evaluate_data[l,:,flux_data_start:-validation_label_size] = flux[train_indices]
            val_evaluate_data[l,:,flux_data_start:-validation_label_size] = flux[val_indices]
            tst_evaluate_data[l,:,flux_data_start:-validation_label_size] = flux[test_indices]

        trn_pre_train_data[:,:,-1] = sim_pre_train[train_indices].T
        val_pre_train_data[:,:,-1] = sim_pre_train[val_indices].T
        tst_pre_train_data[:,:,-1] = sim_pre_train[test_indices].T

        trn_fine_tune_data[:,:,-1] = obs_values[train_indices].T
        val_fine_tune_data[:,:,-1] = obs_values[val_indices].T
        tst_fine_tune_data[:,:,-1] = obs_values[test_indices].T

        trn_evaluate_data[:,:,-2] = sim_values[train_indices].T
        trn_evaluate_data[:,:,-1] = obs_values[train_indices].T
        val_evaluate_data[:,:,-2] = sim_values[val_indices].T
        val_evaluate_data[:,:,-1] = obs_values[val_indices].T
        tst_evaluate_data[:,:,-2] = sim_values[test_indices].T
        tst_evaluate_data[:,:,-1] = obs_values[test_indices].T


        if np.count_nonzero(np.isnan(trn_fine_tune_data[:,:,:-1])) != 0:
            raise Exception("data in trn_fine_tune_data missing")
        if np.count_nonzero(np.isnan(val_fine_tune_data[:,:,:-1])) != 0:
            raise Exception("data in val_fine_tune_data missing")
        if np.count_nonzero(np.isnan(tst_pre_train_data[:,:,:-1])) != 0:
            raise Exception("data in tst_pre_train_data missing")
        if debug:
            print("Put data complete")
            print("-----------------------------------------")

        assert n_dates_train == len(trn_dates)
        assert n_dates_val == len(val_dates)
        assert n_dates_test == len(tst_dates)
        
        #write features and labels to processed data
        print("training: ", first_train_date, "->", last_train_date, "(", n_dates_train, ")")
        print("validation: ", first_val_date, "->", last_val_date, "(", n_dates_val, ")")
        print("testing: ", first_tst_date, "->", last_tst_date, "(", n_dates_test, ")")
        print("-----------------------------------------")
        data_save_path = base_path + f'/processed_extend/seed={seed}/' + name
        # data_save_path = base_path + f'/processed_extend/' + name
        if not os.path.exists(data_save_path): 
            os.makedirs(data_save_path)

        # pre-train data(simulator)
        trn_pre_train_data_path = data_save_path+"/trn_pre_train_data"
        val_pre_train_data_path = data_save_path+"/val_pre_train_data"
        tst_pre_train_data_path = data_save_path+"/tst_pre_train_data"

        np.save(trn_pre_train_data_path, trn_pre_train_data)
        np.save(val_pre_train_data_path, val_pre_train_data)
        np.save(tst_pre_train_data_path, tst_pre_train_data)

        # fine-tune data(observation)
        trn_fine_tune_data_path = data_save_path+"/trn_fine_tune_data"
        val_fine_tune_data_path = data_save_path+"/val_fine_tune_data"
        tst_fine_tune_data_path = data_save_path+"/tst_fine_tune_data"

        np.save(trn_fine_tune_data_path, trn_fine_tune_data)
        np.save(val_fine_tune_data_path, val_fine_tune_data)
        np.save(tst_fine_tune_data_path, tst_fine_tune_data)

        # evaluate data(simulator + observation)
        trn_evaluate_data_path = data_save_path+"/trn_evaluate_data"
        val_evaluate_data_path = data_save_path+"/val_evaluate_data"
        tst_evaluate_data_path = data_save_path+"/tst_evaluate_data"

        np.save(trn_evaluate_data_path, trn_evaluate_data)
        np.save(val_evaluate_data_path, val_evaluate_data)
        np.save(tst_evaluate_data_path, tst_evaluate_data)

        # Date
        trn_dates_path = data_save_path+"/trn_dates"
        val_dates_path = data_save_path+"/val_dates"
        tst_dates_path = data_save_path+"/tst_dates"

        np.save(trn_dates_path, trn_dates)
        np.save(val_dates_path, val_dates)
        np.save(tst_dates_path, tst_dates)

        print(f"{lake_id} completed!")

if __name__ == "__main__":
    COLUMNS_USE = USE_FEATURES_COLUMNS
    for model_index in range(1, 4):
        seed = random_seed[model_index - 1]
        save_npy_data(COLUMNS_USE, seed)