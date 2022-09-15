import numpy as np
import pandas as pd
from collections import defaultdict
from sklearn.preprocessing import StandardScaler
import multiprocessing as mp
import pickle
import os

def get_paths(save_dir, n_cases=-1):
    paths = []
    for root, _, files in os.walk(save_dir):
        for filename in files:
            if len(paths) == n_cases:
                break
            if 'case' in filename:
                paths.append(os.path.join(root, filename))
        if len(paths) == n_cases:
            break
    return paths

def read_datasets(p, paths):
    return pd.read_csv(paths[p], index_col=0)

def collect_raw_data(save_dir, paths):
    
    pool = mp.Pool(mp.cpu_count())
    dfs = pool.starmap(read_datasets, [(p, paths) for p in range(len(paths))])
    pool.close()
    
    return dfs

def generate_input_labels(upstream_turbine_indices, k_delay):
    # n_inputs = (K_DELAY + 1) * ((2 * len(upstream_turbine_indices)) + 1) # delayed axial induction factor and yaw angle for each upstream turbine, delayed freestream wind speed
            
    input_labels = []
    
    for t in upstream_turbine_indices:
        for idx in range(k_delay, -1, -1):
            input_labels.append(f'TurbineWindSpeeds_{t}_minus{idx}')
     
    for t in upstream_turbine_indices:
        for idx in range(k_delay, -1, -1):
            input_labels.append(f'AxIndFactors_{t}_minus{idx}')
            
    for t in upstream_turbine_indices:
        for idx in range(k_delay, -1, -1):
            input_labels.append(f'YawAngles_{t}_minus{idx}')
    
    # for idx in range(k_delay, -1, -1):
    #     input_labels.append(f'FreestreamWindSpeed_minus{idx}')
    
    for idx in range(k_delay, -1, -1):
        input_labels.append(f'FreestreamWindDir_minus{idx}')
    
    return input_labels

def generate_input_vector(case_df, k, upstream_turbine_indices, effective_dk, k_delay):
    inputs = []
    delay_slice = slice(k - (k_delay * effective_dk), k + 1, effective_dk)
    time = case_df.iloc[k:k + 1]['Time'].to_list()
    for t in upstream_turbine_indices:
        inputs = inputs + case_df.loc[delay_slice, f'TurbineWindSpeeds_{t}'].to_list()
        inputs = inputs + case_df.loc[delay_slice, f'AxIndFactors_{t}'].to_list()
        inputs = inputs + case_df.loc[delay_slice, f'YawAngles_{t}'].to_list()
    # inputs = inputs + case_df.loc[delay_slice, f'FreestreamWindSpeed'].to_list()
    inputs = inputs + case_df.loc[delay_slice, f'FreestreamWindDir'].to_list()
    return time, inputs

def split_all_data(model_fi, system_fi, wake_field_dfs, current_input_labels, proportion_training_data, model_type, k_delay, dt):
    X = defaultdict(list)
    y = defaultdict(list)
    time = defaultdict(list)
    measurements_dfs = []

    pool = mp.Pool(mp.cpu_count())
    res = pool.starmap(split_data, [(model_fi, system_fi, case_df, current_input_labels, proportion_training_data, model_type, k_delay, dt) for case_df in wake_field_dfs])
    pool.close()
            
    for time_res, X_res, y_res, measurements_dict_res in res:
        for dataset_type in ['train', 'test', 'full']:
            time[dataset_type] = time[dataset_type] + time_res[dataset_type]
            X[dataset_type].append(X_res[dataset_type])
            y[dataset_type].append(y_res[dataset_type])
            
        measurements_dfs.append(pd.DataFrame(measurements_dict_res))
            
            # for key in measurements_dict_res:
                # measurements_dict[key] = measurements_dict[key] + measurements_dict_res[key]
        
    for dataset_type in ['train', 'test', 'full']:
        time[dataset_type] = np.array(time[dataset_type])
        X[dataset_type] = np.vstack(X[dataset_type])
        y[dataset_type] = np.vstack(y[dataset_type])
    
    return time, X, y, measurements_dfs
 
def split_data(model_fi, system_fi, case_df, current_input_labels, proportion_training_data, model_type, k_delay, dt):
    time = defaultdict(list)
    X = defaultdict(list)
    y = defaultdict(list)
    measurements_dict = {key: [] for key in list(case_df.columns) + ['DatasetType'] + ['IsAdded']}
    
    if model_type == 'error':
        model_fi.floris.farm.flow_field.mean_wind_speed = case_df['FreestreamWindSpeed'].mean()
        model_fi.steady_state_wind = case_df['FreestreamWindSpeed'].mean()
        model_fi.steady_state_wind_direction = case_df['FreestreamWindDir'].mean()
        model_fi.steady_yaw_angles = [case_df[f'YawAngles_{t}'].mean() for t in system_fi.upstream_turbine_indices]
    
    # n_turbines = len(model_fi.floris.farm.turbines) 
    # min_downstream_dist = min(model_fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
    
    training_idx_end = int(np.floor(len(case_df.index) * proportion_training_data))
    # current_input_labels = ['Time', 'FreestreamWindSpeed', 'FreestreamWindDir'] \
    #         + [f'YawAngles_{t}' for t in system_fi.upstream_turbine_indices] \
    #         + [f'AxIndFactors_{t}' for t in system_fi.upstream_turbine_indices] \
    #         + [f'TurbineWindSpeeds_{t}' for t in system_fi.downstream_turbine_indices]
    for idx in range(len(case_df.index)):
        
        dataset_type = 'train' if idx < training_idx_end else 'test'
        row = case_df.iloc[idx]
        
        ## COMPUTE APPROPRIATE SAMPLING ITME, FOR SAME K_DELAY VALUE, CONSIDERING FREESTREAM WIND SPEED (FOR ONLINE LEARNING ONLY)
        downstream_distance = np.array([system_fi.floris.farm.turbine_map.coords[i].x1 - system_fi.min_downstream_dist for i in system_fi.downstream_turbine_indices])
        freestream_ws = row['FreestreamWindSpeed']
        delay_dt = (2 * downstream_distance) * (1 / k_delay) * (1 / freestream_ws)
        effective_dk = int(delay_dt // dt)
        
        if model_type == 'error':
            # sim_time = row['Time']                                                                        
            model_fi.reinitialize_flow_field(# sim_time=sim_time,
                wind_speed=row['FreestreamWindSpeed'], wind_direction=row['FreestreamWindDir'])
            model_fi.calculate_wake(#sim_time=sim_time, 
                    yaw_angles=[row[f'YawAngles_{t}'] for t in system_fi.upstream_turbine_indices], 
                    axial_induction=[row[f'AxIndFactors_{t}'] for t in system_fi.upstream_turbine_indices])
            
            # y_modeled = [model_fi.floris.farm.wind_map.turbine_wind_speed[t] for t in downstream_turbine_indices]
            y_modeled = [model_fi.floris.farm.turbines[t].average_velocity for t in system_fi.downstream_turbine_indices]
        
        for key in current_input_labels:
            measurements_dict[key].append(row[key])
        measurements_dict['DatasetType'].append(dataset_type)
        measurements_dict['IsAdded'].append(False)
         
        # if not enough auto-regressive inputs exist to collect, skip to next time-step in data
        if idx < (k_delay * effective_dk):
            continue
        
        y_measured = [row[f'TurbineWindSpeeds_{t}'] for t in system_fi.downstream_turbine_indices]
        time_input, inputs = generate_input_vector(case_df, idx, system_fi.upstream_turbine_indices, effective_dk, k_delay)
        
        time['full'] = time['full'] + time_input
        time[dataset_type] = time[dataset_type] + time_input
        X['full'].append(inputs)
        X[dataset_type].append(inputs)
        if model_type == 'value':
            y[dataset_type].append(y_measured)
            y['full'].append(y_measured)
        elif model_type == 'error':
            # error in model
            y_err = [y_mod - y_meas for y_mod, y_meas in zip(y_modeled, y_measured)]
            y[dataset_type].append(y_err)
            y['full'].append(y_err)
            
    return time, X, y, measurements_dict


def normalize_data(X, y, renormalize, data_dir):

    if renormalize:
        X_norm = {}
        y_norm = {}
        # exclude Time column
        X_scalar = StandardScaler().fit(X['full'])
        y_scalar = StandardScaler().fit(y['full'])
        
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