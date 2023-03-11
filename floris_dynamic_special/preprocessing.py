import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import pickle
import os

def get_paths(save_dir, df_indices):
    paths = []
    n_total_paths = 0
    for root, _, files in os.walk(save_dir):
        for filename in files:
            print(f'root={root}, filename={filename}')
            if 'case' in filename and 'csv' in filename:
               n_total_paths += 1
   
    print(f'n_total_paths={n_total_paths}')
    if type(df_indices) is int:
        n_cases = df_indices
        df_indices = np.random.randint(n_total_paths, size=(n_cases,))
    elif hasattr(df_indices, '__len__'):
        n_cases = len(df_indices)
    elif df_indices is None:
        n_cases = -1
        
    for root, _, files in os.walk(save_dir):
        for filename in files:
            if len(paths) == n_cases:
                break
            if 'case' in filename and 'csv' in filename and (df_indices is None or
                                                             (int(filename.split('_')[-1].split('.')[0]) in df_indices)):
                paths.append(os.path.join(root, filename))
        if len(paths) == n_cases:
            break
    
    print('No. paths read', len(paths))
    return sorted(paths)

def get_df_paths(df_indices, ts_data_dir, proportion_training_data):

    csv_paths = get_paths(ts_data_dir, df_indices=df_indices)
    # n_training_datasets = int(np.floor(len(csv_paths) * proportion_training_data))
    # full_idx = np.arange(len(csv_paths))
    # np.random.shuffle(full_idx)
    # training_idx = full_idx[:n_training_datasets]
    # testing_idx = full_idx[n_training_datasets:]
    # df_paths = (os.path.join(ts_data_dir, p) for p in csv_paths)
    # wake_field_dfs = collect_raw_data(ts_data_dir, csv_paths)
    # wake_field_dfs = {'train': [wake_field_dfs[i] for i in training_idx],
    #                   'test': [wake_field_dfs[i] for i in testing_idx]}
    return csv_paths

def read_datasets(p, paths):
    return pd.read_csv(paths[p], index_col=0)

def collect_raw_data(save_dir, paths):
    
    pool = mp.Pool(mp.cpu_count())
    dfs = pool.starmap(read_datasets, [(p, paths) for p in range(len(paths))])
    pool.close()
    
    return dfs

def generate_input_labels(upstream_turbine_indices, k_delay):

    input_labels = []
    
    for t in upstream_turbine_indices:
        for idx in range(k_delay, -1, -1):
            input_labels.append(f'TurbineWindSpeeds_{t}_minus{idx}')
     
        for idx in range(k_delay, -1, -1):
            input_labels.append(f'AxIndFactors_{t}_minus{idx}')
            
        for idx in range(k_delay, -1, -1):
            input_labels.append(f'YawAngles_{t}_minus{idx}')

    for idx in range(k_delay, -1, -1):
        input_labels.append(f'FreestreamWindDir_minus{idx}')
    
    return input_labels

def generate_input_vector(case_df, k_idx, upstream_turbine_indices, effective_dk, k_delay):
    inputs = []
    delay_slice = slice(k_idx - (k_delay * effective_dk), k_idx + 1, effective_dk)
    time = [case_df.loc[k_idx]['Time']]
    for t in upstream_turbine_indices:
        inputs = inputs + case_df.loc[delay_slice, f'TurbineWindSpeeds_{t}'].to_list()
        inputs = inputs + case_df.loc[delay_slice, f'AxIndFactors_{t}'].to_list()
        inputs = inputs + case_df.loc[delay_slice, f'YawAngles_{t}'].to_list()
    inputs = inputs + case_df.loc[delay_slice, f'FreestreamWindDir'].to_list()

    return time, inputs

def split_all_data(model_fi, system_fi, wake_field_dfs, current_input_labels, model_type, k_delay, dt):
    X = defaultdict(list)
    y = defaultdict(list)
    time = defaultdict(list)
    # measurements_dfs = {'train': [], 'test': []}

    pool = mp.Pool(mp.cpu_count())
    train_res = pool.starmap(split_data, [(model_fi, system_fi, case_df, current_input_labels, model_type, k_delay, dt,
                                           'train')
                                          for case_df in wake_field_dfs['train']])
    test_res = pool.starmap(split_data, [(model_fi, system_fi, case_df, current_input_labels, model_type, k_delay, dt,
                                          'test')
                                          for case_df in wake_field_dfs['test']])
    pool.close()

    dataset_type = 'train'
    for time_res, X_res, y_res in train_res:
        # for dataset_type in ['train', 'test', 'full']:
        time[dataset_type] = time[dataset_type] + time_res[dataset_type]
        X[dataset_type].append(X_res[dataset_type])
        y[dataset_type].append(y_res[dataset_type])
            
    # measurements_dfs[dataset_type].append(pd.DataFrame(measurements_dict_res))

    dataset_type = 'test'
    for time_res, X_res, y_res in test_res:
        # for dataset_type in ['train', 'test', 'full']:
        time[dataset_type] = time[dataset_type] + time_res[dataset_type]
        X[dataset_type].append(X_res[dataset_type])
        y[dataset_type].append(y_res[dataset_type])

    # measurements_dfs[dataset_type].append(pd.DataFrame(measurements_dict_res))
            
            # for key in measurements_dict_res:
                # measurements_dict[key] = measurements_dict[key] + measurements_dict_res[key]
        
    for dataset_type in ['train', 'test']:
        if len(time[dataset_type]):
            time[dataset_type] = np.array(time[dataset_type])
            X[dataset_type] = np.vstack(X[dataset_type])
            y[dataset_type] = np.vstack(y[dataset_type])
    
    return time, X, y#, measurements_dfs
 
def split_data(model_fi, system_fi, case_df, current_input_labels, model_type, k_delay, dt, dataset_type):
    time = defaultdict(list)
    X = defaultdict(list)
    y = defaultdict(list)
    for idx, row in case_df.iterrows():

        ## COMPUTE APPROPRIATE SAMPLING ITME, FOR SAME K_DELAY VALUE, CONSIDERING FREESTREAM WIND SPEED (FOR ONLINE LEARNING ONLY)
        downstream_distance = np.array([system_fi.floris.farm.turbine_map.coords[i].x1 - system_fi.min_downstream_dist for i in system_fi.downstream_turbine_indices])
        freestream_ws = row['FreestreamWindSpeed']
        delay_dt = (2 * downstream_distance) * (1 / k_delay) * (1 / freestream_ws)
        effective_dk = delay_dt // dt
        # if not enough auto-regressive inputs exist to collect, skip to next time-step in data
        if len(case_df.index) < min(k_delay * effective_dk):
            print('Not enough data points to generate any training data')

        if row['Time'] < min(k_delay * effective_dk):
            continue
        
        y_measured = [row[f'TurbineWindSpeeds_{t}'] for t in system_fi.downstream_turbine_indices]
        
        time_input, inputs = generate_input_vector(case_df, idx, system_fi.upstream_turbine_indices, effective_dk, k_delay)
        
        time[dataset_type] = time[dataset_type] + time_input
        X[dataset_type].append(inputs)
        y[dataset_type].append(y_measured)
            
    return time, X, y


def normalize_data(X, y, renormalize, data_dir):

    if renormalize:
        X_norm = {}
        y_norm = {}
        # exclude Time column
        X_scalar = StandardScaler().fit(X['train'])
        y_scalar = StandardScaler().fit(y['train'])
        
        for data_type in ['train', 'test']:
            X_norm[data_type] = X_scalar.transform(X[data_type])
            y_norm[data_type] = y_scalar.transform(y[data_type])
        
        for var in ['X_norm', 'y_norm', 'X_scalar', 'y_scalar']:
            with open(os.path.join(data_dir, var), 'wb') as f:
                pickle.dump(locals()[var], f)
    
    else:
        for var in ['X_norm', 'y_norm', 'X_scalar', 'y_scalar']:
            with open(os.path.join(data_dir, var), 'rb') as f:
                locals()[var] = pickle.load(f)
    
    return X_norm, y_norm, X_scalar, y_scalar

def add_gaussian_noise(system_fi, measurements, std=0.01):
    noisy_measurements = measurements.copy()
    n_measurements = len(measurements[f'TurbineWindSpeeds_{0}'])
    for ds_idx in system_fi.downstream_turbine_indices:
        noise = 1 + np.random.normal(0, std, size=n_measurements)
        measurements[f'TurbineWindSpeeds_{ds_idx}'] \
            = [measurements[f'TurbineWindSpeeds_{ds_idx}'][i] * noise[i] for i in range(n_measurements)]

    for us_idx in system_fi.upstream_turbine_indices:

        noise = (1 + np.random.normal(0, std, size=n_measurements))
        measurements[f'TurbineWindSpeeds_{us_idx}'] \
            = [measurements[f'TurbineWindSpeeds_{us_idx}'][i] * noise[i] for i in range(n_measurements)]

    return noisy_measurements
