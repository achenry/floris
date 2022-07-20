from floridyn_special.tools.floris_interface import FlorisInterface as DynFlorisInterface
from floris.tools.floris_interface import FlorisInterface as StaticFlorisInterface
from sklearn.datasets import make_friedman2
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import pickle
import multiprocessing as mp

# GOAL: learn propagation speed
# GOAL: changers in ax_ind factor of upstream turbines -> changes in effective wind speed at downstream turbines
# Plug sequence ax ind factor into steady-state model -> velocity deficit at downstream turbines, find time span between ax_ind factor 0 and expected velocity deficit
# Find which ax_ind factors correlates most with effective wind speed, find how std (low covariance) varies with different input dimensions
# Start with constant wind speed and dir
# K_DELAY could vary depending on wind speed; workaround: normalized dt by wind speed, @ 10m/s higher sampling rate than 5m/s => constant number of input delays; divide time index by wind speed e.g. wave number [1/m]

PROPORTION_TRAINING_DATA = 0.6
DT = 1
K_DELAY = 1 # unconservative estimate: (distance to upstream wind turbines / freestream wind speed) / dt, consider half of freestream wind speed for conservative estimate
MODEL_TYPE = 'error'
N_CASES = -1
COLLECT_DATA = True

UPSTREAM_TURBINE_INDICES = [0]
LEARNING_TURBINE_INDEX = 1

def read_datasets(p, paths):
    return pd.read_csv(paths[p], index_col=0)

def generate_input_labels():
    input_labels = []
    for i in UPSTREAM_TURBINE_INDICES:
        for idx in range(K_DELAY, -1, -1):
            input_labels.append(f'AxIndFactors_{i}_minus{idx}')
            input_labels.append(f'YawAngles_{i}_minus{idx}')
    
    for idx in range(K_DELAY, -1, -1):
        input_labels.append(f'FreestreamWindSpeed_minus{idx}')
        input_labels.append(f'FreestreamWindDir_minus{idx}')
    
    return input_labels

def generate_input_vector(case_df, idx):
    inputs = []
    for i in UPSTREAM_TURBINE_INDICES:
        inputs = inputs + case_df.iloc[idx - K_DELAY:idx + 1][f'AxIndFactors_{i}'].to_list()
        inputs = inputs + case_df.iloc[idx - K_DELAY:idx + 1][f'YawAngles_{i}'].to_list()
    inputs = inputs + case_df.iloc[idx - K_DELAY:idx + 1][f'FreestreamWindSpeed'].to_list()
    inputs = inputs + case_df.iloc[idx - K_DELAY:idx + 1][f'FreestreamWindDir'].to_list()
    return np.array(inputs)


def split_data(model_fi, case_df):
    n_turbines = len(model_fi.floris.farm.turbines)
    X = {'train': [], 'test': []}
    y = {'train': [], 'test': []}
    model_fi.floris.farm.flow_field.mean_wind_speed = case_df['FreestreamWindSpeed'].mean()
    model_fi.steady_state_wind = case_df['FreestreamWindSpeed'].mean()
    model_fi.steady_state_wind_direction = case_df['FreestreamWindDir'].mean()
    model_fi.steady_yaw_angles = [case_df[f'YawAngles_{t}'].mean() for t in range(n_turbines)]
    
    training_idx_end = int(np.floor(len(case_df.index) * PROPORTION_TRAINING_DATA))
    for idx in range(len(case_df.index)):
        row = case_df.iloc[idx]
        if MODEL_TYPE == 'error':
            # sim_time = row['Time']
            
            model_fi.reinitialize_flow_field(# sim_time=sim_time,
                wind_speed=row['FreestreamWindSpeed'], wind_direction=row['FreestreamWindDir'])
            model_fi.calculate_wake(#sim_time=sim_time, 
                    yaw_angles=[row[f'YawAngles_{t}'] for t in range(n_turbines)], 
                    axial_induction=[row[f'AxIndFactors_{t}'] for t in range(n_turbines)])
            y_modeled = model_fi.floris.farm.wind_map.turbine_wind_speed[LEARNING_TURBINE_INDEX]
        
        # if not enough auto-regressive inputs exist to collect, skip to next time-step in data 
        if idx < K_DELAY:
            continue
        
        y_measured = row[f'TurbineWindSpeeds_{LEARNING_TURBINE_INDEX}']
        inputs = generate_input_vector(case_df, idx)
        
        dataset_type = 'train' if idx < training_idx_end else 'test'
        
        X[dataset_type].append(inputs)
        if MODEL_TYPE == 'value':
            y[dataset_type].append(y_measured)
        elif MODEL_TYPE == 'error':
            # error in model
            y[dataset_type].append(y_modeled - y_measured)
            
    return X['train'], y['train'], X['test'], y['test']


if __name__ == '__main__':
    
    ## DEFINE PRIOR MODEL
    floris_dir = './2turb_floris_input.json'
    model_fi = DynFlorisInterface(floris_dir)
    n_turbines = len(model_fi.floris.farm.turbines)
    
    ## FETCH RAW DATA
    if sys.platform == 'darwin':
        save_dir = './wake_field_cases'
        data_dir = './data'
        fig_dir = './figs'
    elif sys.platform == 'linux':
        save_dir = '/scratch/ahenry/wake_field_cases'
        data_dir = '/scratch/ahenry/data'
        fig_dir = '/scratch/ahenry/figs'
    
    for dir in [data_dir, fig_dir]:
        if not os.path.exists(dir):
            os.mkdir(dir)
    
    paths = []
    for root, dir, files in os.walk(save_dir):
        for filename in files:
            if len(paths) == N_CASES:
                break
            if 'case' in filename:
                paths.append(os.path.join(root, filename))
        if len(paths) == N_CASES:
            break
        
    # COMPILE TRAINING AND TESTING DATA
    n_inputs = (K_DELAY + 1) * ((2 * len(UPSTREAM_TURBINE_INDICES)) + 1) # delayed axial induction factor and yaw angle for each upstream turbine, delayed freestream wind speed
    input_labels = generate_input_labels()
        
    if COLLECT_DATA:
        pool = mp.Pool(mp.cpu_count())
        wake_field_dfs = pool.starmap(read_datasets, [(p, paths) for p in range(len(paths))])

        X = {'train': [], 'test': []}
        y = {'train': [], 'test': []}
        
        pool = mp.Pool(mp.cpu_count())
        res = pool.starmap(split_data, [(model_fi, case_df) for case_df in wake_field_dfs])
        
        for X_train, y_train, X_test, y_test in res:
            X['train'].append(X_train)
            y['train'] = y['train'] + y_train
            X['test'].append(X_test)
            y['test'] = y['test'] + y_test
         
        for dataset_type in ['train', 'test']:
            X[dataset_type] = np.vstack(X[dataset_type])
            y[dataset_type] = np.array(y[dataset_type])

        with open(os.path.join(data_dir, 'X'), 'wb') as f:
            pickle.dump(X, f)
        
        with open(os.path.join(data_dir, 'y'), 'wb') as f:
            pickle.dump(y, f)
    else:
        with open(os.path.join(data_dir, 'X'), 'rb') as f:
            X = pickle.load(f)
            
        with open(os.path.join(data_dir, 'y'), 'rb') as f:
            y = pickle.load(f)
            
    ## INSPECT DATA
    if False:
        fig, axs = plt.subplots(int(len(wake_field_dfs[0].columns[1:]) // 2), 2, sharex=True)
        axs = axs.flatten()
        for df in wake_field_dfs:
            time = df['Time'].to_numpy()
            for state_idx, col in enumerate(df.columns[1:]):
                axs[state_idx].plot(time, df[col].to_numpy())
                axs[state_idx].set(ylabel=col)
            axs[-1].set(xlabel='Time [s]')
        plt.show()
        
    ## PARAMETERIZE GAUSSIAN PROCESS REGRESSOR
    kernel = RBF() # DotProduct() + WhiteKernel()
    gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True, n_restarts_optimizer=10, 
                                   random_state=0).fit(X['train'], y['train'])
    for dataset_type in ['train', 'test']:
        print(f'Score on {dataset_type} Data:', gpr.score(X[dataset_type], y[dataset_type]), sep=' ') # test on training/testing data
    
    ## PLOT TRAINING/TEST DATA AND GP PREDICTION
    fig, ax = plt.subplots(1, 2)
    mean = {'train': None, 'test': None}
    std = {'train': None, 'test': None}
    field_idx = input_labels.index(f'FreestreamWindSpeed_minus0')
    sort_idx = {'train': None, 'test': None}
    
    for d, dataset_type in enumerate(['train', 'test']):
        mean[dataset_type], std[dataset_type] = gpr.predict(X[dataset_type], return_std=True)
        sort_idx[dataset_type] = np.argsort(X[dataset_type][:, field_idx])

        ax[d].plot(X[dataset_type][sort_idx[dataset_type], field_idx], mean[dataset_type][sort_idx[dataset_type]], label='Predicted')
        ax[d].fill_between(X[dataset_type][sort_idx[dataset_type], field_idx], 
                           mean[dataset_type][sort_idx[dataset_type]] - std[dataset_type][sort_idx[dataset_type]], 
                           mean[dataset_type][sort_idx[dataset_type]] + std[dataset_type][sort_idx[dataset_type]], 
                           alpha=0.1, label='Std. Dev')
        ax[d].scatter(X[dataset_type][sort_idx[dataset_type], field_idx], y[dataset_type][sort_idx[dataset_type]], linestyle='dashed', color='black', label='Measurements')
        ax[d].set(title=f'Turbine {LEARNING_TURBINE_INDEX} Wind Speed [m/s]', xlabel=f'X[{dataset_type}][:, {field_idx}]', xlim=[min(X[dataset_type][:, field_idx]), max(X[dataset_type][:, field_idx])])
    
    plt.show()
    plt.savefig(os.path.join(fig_dir, 'training_test_prediction.png'))
    
    ## MODEL WAKE-FIELD DATA AT TURBINES AND COMPARE TO GP PREDICTIONS
    
    test_case_idx = 0
    if not COLLECT_DATA:
       test_case_df = read_datasets(test_case_idx, paths)
    else:
        test_case_df = wake_field_dfs[test_case_idx]
        
    y_true = []
    y_pred = []
    y_std = []
    time = test_case_df['Time']
    
    sim_fi = DynFlorisInterface(floris_dir)
    sim_fi.floris.farm.flow_field.mean_wind_speed = test_case_df['FreestreamWindSpeed'].mean()
    model_fi.floris.farm.flow_field.mean_wind_speed = test_case_df['FreestreamWindSpeed'].mean()
    for row_idx in range(len(test_case_df.index)):
        row = test_case_df.iloc[row_idx]
        sim_time = row['Time']
        
        sim_fi.reinitialize_flow_field(wind_speed=row['FreestreamWindSpeed'], wind_direction=row['FreestreamWindDir'], sim_time=sim_time)
        
        sim_fi.calculate_wake(sim_time=sim_time, 
                          yaw_angles=[row[f'YawAngles_{t}'] for t in range(n_turbines)], 
                          axial_induction=[row[f'AxIndFactors_{t}'] for t in range(n_turbines)])
        
        if MODEL_TYPE == 'error':
            
            model_fi.reinitialize_flow_field(wind_speed=row['FreestreamWindSpeed'], wind_direction=row['FreestreamWindDir'])
            model_fi.calculate_wake(yaw_angles=[row[f'YawAngles_{t}'] for t in range(n_turbines)], 
                                    axial_induction=[row[f'AxIndFactors_{t}'] for t in range(n_turbines)])
            y_modeled = model_fi.floris.farm.wind_map.turbine_wind_speed[LEARNING_TURBINE_INDEX]
                          
        if row_idx >= K_DELAY:
            y_true.append(sim_fi.floris.farm.wind_map.turbine_wind_speed[LEARNING_TURBINE_INDEX])
            
            X_ip = np.array(generate_input_vector(test_case_df, row_idx))[np.newaxis, :]
            mean, std = gpr.predict(X_ip, return_std=True)
            
            if MODEL_TYPE == 'value':
                y_pred.append(mean)
            elif MODEL_TYPE == 'error':
                y_pred.append(y_modeled + mean)
                
            y_std.append(std)
    
    y_pred = np.concatenate(y_pred)
    y_std = np.concatenate(y_std)
    
    fig, ax = plt.subplots(1, 1)
    ax.set(title=f'Turbine {LEARNING_TURBINE_INDEX} Wind Speed [m/s]', xlabel='Time [s]')
    ax.plot(time[K_DELAY:], y_true, label='True')
    ax.plot(time[K_DELAY:], y_pred, label='Predicted Mean')
    ax.fill_between(time[K_DELAY:], y_pred - y_std, y_pred + y_std, alpha=0.1, label='Predicted Std. Dev.')
    ax.legend()
    plt.show()
    plt.savefig(os.path.join(fig_dir, 'true_vs_predicted_sim.png'))