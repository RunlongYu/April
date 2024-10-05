import time
import os
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from collections import Counter
import numpy as np
import hashlib
from collections import defaultdict


DROP_COLUMNS = ['AF', 'geometry', 'Hylak_id', 'Grand_id', 'Lake_name', 'Country', 'Continent',
                'Hylak_id', 'Lake_type', 'Grand_id', 'n_epi', 'n_hypo', 'n_mixed', 'fit_train', 'fit_test',
                   'fit_all', 'obs_total', 'mean_prob_dys', 'var_prob_dys', 'mean_prob_eumixo',
                   'var_prob_eumixo', 'mean_prob_oligo', 'var_prob_oligo', 'Poly_src',
                'NEP_mgm3d', 'SED_mgm2d', 'MIN_mgm3d', 'khalf', 'Pour_long', 'Pour_lat', 'Lake_area', 'Shore_dev',
                'ct', 'categorical_ts']


USE_FEATURES_COLUMNS = ['sat_hypo', 'thermocline_depth',
       'temperature_epi', 'temperature_hypo', 'volume_epi', 'volume_hypo',
       'wind', 'airtemp', 'fnep', 'fmineral', 'fsed', 'fatm', 'fdiff',
       'fentr_epi', 'fentr_hyp', 'eutro', 'oligo', 'dys', 'water', 'developed',
       'barren', 'forest', 'shrubland', 'herbaceous', 'cultivated', 'wetlands',
       'depth', 'area', 'elev', 'Shore_len', 'Vol_total', 'Vol_res', 'Vol_src',
       'Depth_avg', 'Dis_avg', 'Res_time', 'Elevation', 'Slope_100',
       'Wshd_area']

USE_FEATURES_COLUMNS_LAYER = ['sat_hypo', 'thermocline_depth',
       'temperature_epi', 'temperature_hypo', 'volume_epi', 'volume_hypo',
       'wind', 'airtemp', 'fnep', 'fmineral', 'fsed', 'fatm', 'fdiff',
       'fentr_epi', 'fentr_hyp', 'eutro', 'oligo', 'dys', 'water', 'developed',
       'barren', 'forest', 'shrubland', 'herbaceous', 'cultivated', 'wetlands',
       'depth', 'area', 'elev', 'Shore_len', 'Vol_total', 'Vol_res', 'Vol_src',
       'Depth_avg', 'Dis_avg', 'Res_time', 'Elevation', 'Slope_100',
       'Wshd_area','mixed','layer']

FLUX_COLUMNS = ['volume_epi','volume_hypo','fnep', 'fmineral', 'fsed', 'fatm', 'fdiff', 'fentr_epi', 'fentr_hyp', 'extend']

USE_FEATURES_COLUMNS_NOFLUX = ['sat_hypo', 'thermocline_depth',
       'temperature_epi', 'temperature_hypo', 'volume_epi', 'volume_hypo',
       'wind', 'airtemp', 'eutro', 'oligo', 'dys', 'water', 'developed',
       'barren', 'forest', 'shrubland', 'herbaceous', 'cultivated', 'wetlands',
       'depth', 'area', 'elev', 'Shore_len', 'Vol_total', 'Vol_res', 'Vol_src',
       'Depth_avg', 'Dis_avg', 'Res_time', 'Elevation', 'Slope_100',
       'Wshd_area']

USE_FEATURES_COLUMNS2 = ['sat_hypo', 'thermocline_depth',
       'temperature_epi', 'temperature_hypo', 'volume_epi', 'volume_hypo',
       'wind', 'airtemp', 'fnep', 'fmineral', 'fsed', 'fatm', 'fdiff',
       'fentr_epi', 'eutro', 'oligo', 'dys', 'water', 'developed',
       'barren', 'forest', 'shrubland', 'herbaceous', 'cultivated', 'wetlands',
       'depth', 'area', 'Shore_len', 'Vol_total', 'Vol_res', 'Vol_src',
       'Depth_avg', 'Dis_avg', 'Res_time', 'Elevation', 'Slope_100',
       'Wshd_area']

FLUX_START = -1-len(USE_FEATURES_COLUMNS_LAYER)