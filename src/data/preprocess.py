from lakePreprocess import USE_FEATURES_COLUMNS, USE_FEATURES_COLUMNS_LAYER, FLUX_COLUMNS
import pandas as pd
from pytorch_data_operations import buildManyLakeDataByIds
from sklearn.preprocessing import StandardScaler
from pathlib import Path
from sklearn.cluster import KMeans
import numpy as np
from tqdm import tqdm
import os
import re

def pre_geometry():
    # 'pattern' is a regex pattern used for extracting coordinates
    pattern = r'c\(([^,]+),\s*([^)]+)\)'

    # Base path for data files
    base_path = '../../data'
    read_path = os.path.join(base_path, 'raw')
    output_path = os.path.join(base_path, 'utils/group_info.csv')
    new_ids_output_path = os.path.join(base_path, 'utils/ids.csv')
    ids = pd.read_csv('../../data/utils/ids_total.csv')

    # Initialize lists to store extracted data
    geometries = []
    lake_ids = []
    depths = []
    area = []

    # Iterate over each lake ID from the loaded ids file
    print("Processing geometry info...")
    for lake_id in tqdm(ids['nhdhr_id']):
        nid = str(lake_id)
        file_path = os.path.join(read_path, nid + '.csv')

        # Read specific columns needed from each lake's CSV file
        geometry_df = pd.read_csv(file_path, usecols=['nhdhr_id', 'geometry', 'Depth_avg', 'area'])

        # Process each row to find the first valid 'geometry' and extract coordinate values
        for index, row in geometry_df.iterrows():
            geom = row['geometry']
            if pd.notna(geom) and geom != 'c(NA, NA)':
                matches = re.search(pattern, geom)
                if matches:
                    # If a valid geometry is found, extract and convert the coordinates to floats
                    x, y = map(float, matches.groups())
                    geometries.append([x, y])
                    lake_ids.append(row['nhdhr_id'])
                    area.append(geometry_df['area'][0])
                    # Search for the first non-NA average depth and break after finding
                    for index, row in geometry_df.iterrows():
                        depth = row['Depth_avg']
                        if pd.notna(depth):
                            depths.append(depth)
                            break
                    break
        
    # Create a DataFrame to store the extracted data
    output_df = pd.DataFrame({
        'nhdhr_id': lake_ids,
        'geometry': [f'{x};{y}' for x, y in geometries],
        'Depth_avg': depths,
        'area': area
    })

    # Save the DataFrame to a CSV file
    output_df.to_csv(output_path, index=False)
    output_df['nhdhr_id'].to_csv(new_ids_output_path, index = False)

    print(f'Save geo information to {output_path}')
    print(f'Save ids(lakesa have geo location) to f{new_ids_output_path}')



def grouping():
    print("==========================")
    random_seed = 42
    df = pd.read_csv('../../data/utils/group_info.csv')
    base_path = '../../data'
    # Apply logarithm to area and volume
    df['log_area'] = np.log(df['area']) / np.log(2)
    df['log_Depth_avg'] = np.log(df['Depth_avg']) / np.log(2)

    # First round of KMeans clustering
    initial_kmeans = KMeans(n_clusters=2, random_state=random_seed)
    df['cluster'] = initial_kmeans.fit_predict(df[['log_area', 'log_Depth_avg']])

    def re_cluster(df, parent_cluster_label, n_clusters, offset, random_seed):
        cluster_data = df[df['cluster'] == parent_cluster_label]
        
        re_cluster_kmeans = KMeans(n_clusters=n_clusters, random_state=random_seed)
        re_cluster_labels = re_cluster_kmeans.fit_predict(cluster_data[['log_area', 'log_Depth_avg']])
        new_labels = offset + re_cluster_labels
        df.loc[cluster_data.index, 'cluster'] = new_labels

    # Second round of clustering on the largest cluster from the first round
    largest_cluster = df['cluster'].value_counts().idxmax()
    re_cluster(df, largest_cluster, 2, offset=2, random_seed=random_seed)

    # Find new largest cluster for the third round
    # Exclude previously re-clustered clusters (offset >= 2)
    third_round_candidates = df[df['cluster'] > 0]
    third_largest_cluster = third_round_candidates['cluster'].value_counts().idxmax()
    re_cluster(df, third_largest_cluster, 2, offset=4, random_seed=random_seed)

    output_path = os.path.join(base_path, 'utils/groups/vol_area')
    os.makedirs(output_path, exist_ok=True)  # Create the directory if it doesn't exist

    cluster_num = df['cluster'].value_counts().shape[0]
    for i in range(cluster_num):
        cluster_id = df['cluster'].value_counts().index[i]
        cluster_data = df[df['cluster'] == cluster_id]
        cluster_ids = cluster_data
        output_file = os.path.join(output_path, f'cluster_{i + 1}.csv')
        df.loc[df['cluster'] == cluster_id, 'cluster'] = -(i + 1)
        cluster_ids['cluster'] = (i + 1)
        cluster_ids.to_csv(output_file, index=False)

    save_path = '../../data/utils/ids.csv'
    df['cluster'] = -df['cluster']
    df[['nhdhr_id', 'cluster']].to_csv(save_path, index=False)
    print("Finish saving different cluster IDs into their respective CSV files.")

def save_npy_data(COLUMNS_USE):
    debug = False
    print("==========================")
    print("Create numpy files for all lakes(have longtitude and latitude information)")
    my_path = os.path.abspath(os.path.dirname(__file__))
    base_path = my_path+ '/../../data'
    ids = pd.read_csv('../../data/utils/ids.csv') # 425 lakes
    n_features = len(COLUMNS_USE) + 1 # +1 represent "mixed"

    n_flux = len(FLUX_COLUMNS)
    raw_data_path = base_path + '/raw/'
    read_path = base_path + '/norm/'
    for number, lake_id in tqdm(enumerate(ids['nhdhr_id'])):
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
        flux = flux_df.to_numpy()

        assert not np.isnan(flux).any(), "Flux data contains NaN values after preprocessing"

        if debug:
            print("flux shape:", flux.shape)
        if debug:
            print("features nan:",np.count_nonzero(np.isnan(features)))
            print("features not nan", np.count_nonzero(~np.isnan(features)))
            
        features_processed = features
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
        if debug:
            print("training: ", first_train_date, "->", last_train_date, "(", n_dates_train, ")")
            print("validation: ", first_val_date, "->", last_val_date, "(", n_dates_val, ")")
            print("testing: ", first_tst_date, "->", last_tst_date, "(", n_dates_test, ")")
            print("-----------------------------------------")
        data_save_path = base_path + f'/processed/' + name
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
        if debug:
            print(f"{lake_id} completed!")

if __name__ == "__main__":
    COLUMNS_USE = USE_FEATURES_COLUMNS
    save_npy_data(COLUMNS_USE)