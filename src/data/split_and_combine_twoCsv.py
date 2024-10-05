import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from lakePreprocess import USE_FEATURES_COLUMNS
from sklearn.preprocessing import StandardScaler

def calculate_mean_std(noise_list, raw_data):
    feature_df = raw_data
    # Create an empty DataFrame to store results
    results = pd.DataFrame()

    # Iterate over the feature list and calculate the mean and standard deviation for each feature
    for feature in noise_list:
        mean_value = feature_df[feature].mean()  # Calculate mean
        std_value = feature_df[feature].std()    # Calculate standard deviation
        # Prepare a row with the feature, its mean, and standard deviation
        result_row = pd.DataFrame({
            'feature': [feature],
            'mean': [mean_value],
            'std': [std_value]
        })
        # Append the result row to the results DataFrame
        results = pd.concat([results, result_row], ignore_index=True)
    return results

def fill_csv(feature_df, mean_std):
# Delete first day of 1979
    # feature_df = feature_df.iloc[1:,:].copy()
    feature_df = feature_df
    # Morphometric_list    
    Morphometric_list = ['Shore_len', 'Vol_total', 'Vol_res', 'Vol_src','Depth_avg', 
    'Dis_avg', 'Res_time', 'Elevation', 'Slope_100','Wshd_area']
    for feature in Morphometric_list:
        missing_count = feature_df[feature].isna().sum()
        if feature_df[feature].isna().sum() == len(feature_df[feature]):
            mean_value = mean_std.loc[mean_std['feature'] == feature, 'mean'].values[0]
            feature_df.loc[feature_df[feature].isna(), feature] = mean_value

        elif missing_count > 0:
            feature_df[feature].fillna(feature_df[feature].mean(), inplace=True)

    # Flux data
    flux1 = ['fnep', 'fmineral', 'fsed', 'fatm', 'fdiff']
    for flux in flux1:
        feature_df[flux].fillna(method='bfill', inplace=True)

    flux2 = ['fentr_epi', 'fentr_hyp']
    for flux in flux2:
        feature_df[flux].fillna(0, inplace=True)
    
    #Land use
    lanuse_list = ['water', 'developed',
    'barren', 'forest', 'shrubland', 'herbaceous', 'cultivated', 'wetlands']
    for feature in lanuse_list:
        missing_count = feature_df[feature].isna().sum()
        if feature_df[feature].isna().sum() == len(feature_df[feature]):
            mean_value = mean_std.loc[mean_std['feature'] == feature, 'mean'].values[0]
            feature_df.loc[feature_df[feature].isna(), feature] = mean_value

        elif missing_count > 0:
            feature_df[feature].fillna(feature_df[feature].mean(), inplace=True)

    # Trophic State and Classification
    noise_list = ['eutro', 'oligo', 'dys', 'sat_hypo', 'thermocline_depth']
    for feature in noise_list:
        mean_value = feature_df[feature].mean()
        std_value = feature_df[feature].std()
        missing_count = feature_df[feature].isna().sum()

        if feature_df[feature].isna().sum() == len(feature_df[feature]):
            mean_value = mean_std.loc[mean_std['feature'] == feature, 'mean'].values[0]
            std_value = mean_std.loc[mean_std['feature'] == feature, 'std'].values[0]
            noise = np.random.normal(loc=mean_value, scale=std_value, size=missing_count)
            feature_df.loc[feature_df[feature].isna(), feature] = noise

        elif missing_count > 0:
            noise = np.random.normal(loc=mean_value, scale=std_value, size=missing_count)
            feature_df.loc[feature_df[feature].isna(), feature] = noise

    temperature_list = ['temperature_epi', 'temperature_hypo']
    # compare temperature_epi temperature_hypo nan
    mask1 = feature_df[temperature_list[0]].isna()
    mask2 = feature_df[temperature_list[1]].isna()
    if not mask1.equals(mask2):
        raise Exception("temperature_epi and temperature_hypo not match")
    
    assert not feature_df['temperature_total'].isna().any(), "temperature_total column contains NaN values"
    # Ensure that temperature columns have matched missing values
    for feature in temperature_list:
        feature_df[feature] = feature_df[feature].fillna(feature_df['temperature_total'])

    assert not feature_df['temperature_epi'].isna().any(), "temperature_epi column contains NaN values"
    feature_df['volume_epi'].fillna(feature_df['volume_total'], inplace=True)
    feature_df['volume_hypo'].fillna(0, inplace=True)


    # Handle observed and simulated data overlap
    overlap_condition = (~feature_df['obs_tot'].isna()) & (~feature_df['obs_epi'].isna())
    if overlap_condition.any():
        raise ValueError("Error: Overlap detected between 'obs_tot' and 'obs_epi'.")

    # put obs_total in obs_epi
    feature_df.loc[~feature_df['obs_tot'].isna(), 'obs_epi'] = feature_df['obs_tot']

     # Create a flag indicating if the data is mixed or not based on simulation values
    mixed_condition = (feature_df['sim_epi'] == feature_df['sim_hyp']) & (feature_df['sim_hyp'] == feature_df['sim_tot'])          
    feature_df['mixed'] = (mixed_condition).astype(int)     

    # extend
    feature_df['extend'] = 0
 
    # when mixed, set sim_hypo = nan
    # feature_df.loc[mixed_condition, 'sim_hyp'] = np.nan

    for feature in USE_FEATURES_COLUMNS:
        all_not_na = feature_df[feature].notna().all()
        if not all_not_na.all():
            raise Exception(f"Data missing in {feature}")
        
    return feature_df

def combine_and_split_csv():
    print("==========================")
    print("Reading raw csv flies...")
    print("This will take a few minutes.")
    # Define the path to the raw data file
    raw_path = '../../data/old_lake_data_all.csv'
    # Read the CSV file and remove duplicate entries based on specific columns
    raw_data = pd.read_csv(raw_path)
    raw_data = raw_data.drop_duplicates(subset=['nhdhr_id', 'datetime'])
    ids = raw_data['nhdhr_id'].drop_duplicates()
    ids_dir = '../../data/utils'
    os.makedirs(ids_dir, exist_ok=True)  # Creates the directory if it does not exist
    # Save the unique ids to a CSV file in the specified directory
    ids_path = os.path.join(ids_dir, 'ids_total.csv')
    ids.to_csv(ids_path, index=False)
    # Define the path to the new features data file
    new_feature_path = '../../data/raw_data-allseasons.csv'
    # Read selected columns from the new features CSV file
    new_features = pd.read_csv(new_feature_path, usecols=['lake', 'datetime', 'volume_epi', 'volume_hypo', 'volume_total', 'temperature_total'])
    print("Finish reading raw csv flies")
    print("==========================")
    # Print shapes of the dataframes for debugging
    print("raw_data shape:", raw_data.shape)
    print("new_features shape:", new_features.shape)
    print("==========================")
    # Ensure there are no NaN values in critical columns and that the data shapes match
    assert not new_features['temperature_total'].isna().any(), "temperature_total column contains NaN values"
    assert new_features.shape[0] == raw_data.shape[0], "data shape not match"

    # Reset index of the raw data
    raw_data.reset_index(drop=False, inplace=True)

    # Insert new columns from new features into the raw data at specific positions
    raw_data.insert(13, 'temperature_total', new_features['temperature_total'])
    raw_data.insert(16, 'volume_total', new_features['volume_total'])

    # Define a list of columns for noise calculation
    noise_list = ['eutro', 'oligo', 'dys', 'water', 'developed', 'barren', 'forest', 'shrubland', 'herbaceous', 'cultivated', 'wetlands', 'Shore_len', 'Vol_total', 'Vol_res', 'Vol_src','Depth_avg', 'Dis_avg', 'Res_time', 'Elevation', 'Slope_100','Wshd_area']
    # Calculate mean and standard deviation for specified columns
    mean_std = calculate_mean_std(noise_list, raw_data)
    # print(mean_std)

    # Group the data by lake ID
    raw_grouped = raw_data.groupby('nhdhr_id')
    filled_data_list = []
    print("Fliing nan data(This will take a few minutes)")
    # Fill missing data for each group and ensure there are no NaN values in temperature_total
    for lake_id, group in tqdm(raw_grouped):
            assert not group['temperature_total'].isna().any(), "group temperature_total column contains NaN values"
            filled_group = fill_csv(group, mean_std)
            filled_data_list.append(filled_group)

    # Concatenate all filled groups into a single dataframe
    filled_data = pd.concat(filled_data_list, ignore_index=True)
    nrom_data = filled_data.copy()

    # Initialize the StandardScaler
    scaler = StandardScaler()

    # Standardize feature columns
    features = nrom_data[USE_FEATURES_COLUMNS]
    features_processed = scaler.fit_transform(features)
    nrom_data[USE_FEATURES_COLUMNS] = features_processed

    # Define and create directories for saving raw and normalized data
    output_dir = '../../data/raw'
    os.makedirs(output_dir, exist_ok=True)
    norm_output_dir = '../../data/norm'
    os.makedirs(norm_output_dir, exist_ok=True)

    # Group and save the processed data to CSV files for each lake, handling duplicates in datetime
    raw_filled_grouped = filled_data.groupby('nhdhr_id')
    norm_filled_grouped = nrom_data.groupby('nhdhr_id')

    print("Finish filling nan data(This will take a few minutes)")
    print("==========================")
    print("Save unnormalized data to CSV for each lake(This will take a few minutes)")
    print("--------")
    for lake_id, group in tqdm(raw_filled_grouped):
        group = group.drop_duplicates(subset=['datetime'])
        group = group.iloc[1:]  # skip first row if required
        output_file_path = os.path.join(output_dir, f'{lake_id}.csv')
        group.to_csv(output_file_path, index=False)
    print("Save ormalized data to CSV for each lake(This will take a few minutes)")
    print("--------")
    for lake_id, group in tqdm(norm_filled_grouped):
        group = group.drop_duplicates(subset=['datetime'])
        group = group.iloc[1:]  # skip first row if required
        output_file_path = os.path.join(norm_output_dir, f'{lake_id}.csv')
        group.to_csv(output_file_path, index=False)

    print("CSV files for all lakes have been processed and saved successfully.")