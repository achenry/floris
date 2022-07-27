from floridyn_special.tools.floris_interface import FlorisInterface as DynFlorisInterface
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import DotProduct, WhiteKernel, RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os
import sys
import pickle
import multiprocessing as mp
from collections import defaultdict
from scipy.optimize import minimize

# GOAL: learn propagation speed
# GOAL: changers in ax_ind factor of upstream turbines -> changes in effective wind speed at downstream turbines
# Plug sequence ax ind factor into steady-state model -> velocity deficit at downstream turbines, find time span between ax_ind factor 0 and expected velocity deficit
# Find which ax_ind factors correlates most with effective wind speed, find how std (low covariance) varies with different input dimensions
# Start with constant wind speed and dir
# K_DELAY could vary depending on wind speed; workaround: normalized dt by wind speed, @ 10m/s higher sampling rate than 5m/s => constant number of input delays; divide time index by wind speed e.g. wave number [1/m]

PROPORTION_TRAINING_DATA = 0.85
DT = 1
K_DELAY = 10 # unconservative estimate: (distance to upstream wind turbines / freestream wind speed) / dt, consider half of freestream wind speed for conservative estimate
MODEL_TYPE = 'error'
N_CASES = -1
COLLECT_DATA = True
NORMALIZE_DATA = False
PLOT_DATA = True

def optimizer(fun, initial_theta, bounds):
    res = minimize(fun, initial_theta, 
                   method="L-BFGS-B", jac=True, 
                   bounds=bounds, 
                   options={'maxiter': 1000})
    theta_opt = res.x
    func_min = res.fun
    return theta_opt, func_min

def read_datasets(p, paths):
    return pd.read_csv(paths[p], index_col=0)

def generate_input_labels(upstream_turbine_indices):
    input_labels = []
    
    input_labels.append('Time')
    
    for t in upstream_turbine_indices:
        for idx in range(K_DELAY, -1, -1):
            input_labels.append(f'AxIndFactors_{t}_minus{idx}')
            
    for t in upstream_turbine_indices:
        for idx in range(K_DELAY, -1, -1):
            input_labels.append(f'YawAngles_{t}_minus{idx}')
    
    for idx in range(K_DELAY, -1, -1):
        input_labels.append(f'FreestreamWindSpeed_minus{idx}')
    
    for idx in range(K_DELAY, -1, -1):
        input_labels.append(f'FreestreamWindDir_minus{idx}')
    
    return input_labels

def generate_input_vector(case_df, idx, upstream_turbine_indices, effective_dk):
    inputs = []
    delay_slice = slice(idx - (K_DELAY * effective_dk), idx + 1, effective_dk)
    inputs = inputs + case_df.iloc[idx:idx + 1]['Time'].to_list()
    for t in upstream_turbine_indices:
        inputs = inputs + case_df.iloc[delay_slice][f'AxIndFactors_{t}'].to_list()
        inputs = inputs + case_df.iloc[delay_slice][f'YawAngles_{t}'].to_list()
    inputs = inputs + case_df.iloc[delay_slice][f'FreestreamWindSpeed'].to_list()
    inputs = inputs + case_df.iloc[delay_slice][f'FreestreamWindDir'].to_list()
    return np.array(inputs)


def split_data(model_fi, case_df, upstream_turbine_indices, downstream_turbine_indices):
    X = defaultdict(list)
    y = defaultdict(list)
    model_fi.floris.farm.flow_field.mean_wind_speed = case_df['FreestreamWindSpeed'].mean()
    model_fi.steady_state_wind = case_df['FreestreamWindSpeed'].mean()
    model_fi.steady_state_wind_direction = case_df['FreestreamWindDir'].mean()
    model_fi.steady_yaw_angles = [case_df[f'YawAngles_{t}'].mean() for t in upstream_turbine_indices]
    
    n_turbines = len(model_fi.floris.farm.turbines) 
    min_downstream_dist = min(model_fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
    
    training_idx_end = int(np.floor(len(case_df.index) * PROPORTION_TRAINING_DATA))
    for idx in range(len(case_df.index)):
        row = case_df.iloc[idx]
        
        ## COMPUTE APPROPRIATE SAMPLING ITME, FOR SAME K_DELAY VALUE, CONSIDERING FREESTREAM WIND SPEED (FOR ONLINE LEARNING ONLY)
        downstream_distance = np.array([model_fi.floris.farm.turbine_map.coords[i].x1 - min_downstream_dist for i in downstream_turbine_indices])
        freestream_ws = row['FreestreamWindSpeed']
        delay_dt = (2 * downstream_distance) * (1 / K_DELAY) * (1 / freestream_ws)
        effective_dk = int(delay_dt // DT)
        
        if MODEL_TYPE == 'error':
            # sim_time = row['Time']                                                                        
            model_fi.reinitialize_flow_field(# sim_time=sim_time,
                wind_speed=row['FreestreamWindSpeed'], wind_direction=row['FreestreamWindDir'])
            model_fi.calculate_wake(#sim_time=sim_time, 
                    yaw_angles=[row[f'YawAngles_{t}'] for t in upstream_turbine_indices], 
                    axial_induction=[row[f'AxIndFactors_{t}'] for t in upstream_turbine_indices])
            
            # y_modeled = [model_fi.floris.farm.wind_map.turbine_wind_speed[t] for t in downstream_turbine_indices]
            y_modeled = [model_fi.floris.farm.turbines[t].average_velocity for t in downstream_turbine_indices]
        
        # if not enough auto-regressive inputs exist to collect, skip to next time-step in data
        if idx < (K_DELAY * effective_dk):
            continue
        
        y_measured = [row[f'TurbineWindSpeeds_{t}'] for t in downstream_turbine_indices]
        inputs = generate_input_vector(case_df, idx, upstream_turbine_indices, effective_dk)
        
        dataset_type = 'train' if idx < training_idx_end else 'test'
        
        X['full'].append(inputs)
        X[dataset_type].append(inputs)
        if MODEL_TYPE == 'value':
            y[dataset_type].append(y_measured)
            y['full'].append(y_measured)
        elif MODEL_TYPE == 'error':
            # error in model
            y_err = [y_mod - y_meas for y_mod, y_meas in zip(y_modeled, y_measured)]
            y[dataset_type].append(y_err)
            y['full'].append(y_err)
            
    return X['train'], y['train'], X['test'], y['test'], X['full'], y['full']


if __name__ == '__main__':
    
    ## DEFINE PRIOR MODEL
    floris_dir = './2turb_floris_input.json'
    model_fi = DynFlorisInterface(floris_dir)
    n_turbines = len(model_fi.floris.farm.turbines)
    
    ## FETCH RAW DATA
    if sys.platform == 'darwin':
        save_dir = './2turb_wake_field_cases'
        data_dir = './data'
        fig_dir = './figs'
    elif sys.platform == 'linux':
        save_dir = '/scratch/ahenry/2turb_wake_field_cases'
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
        
    max_downstream_dist = max(model_fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
    min_downstream_dist = min(model_fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
    # exclude most downstream turbine
    upstream_turbine_indices = [t for t in range(n_turbines) if model_fi.floris.farm.turbine_map.coords[t].x1 < max_downstream_dist]
    n_upstream_turbines = len(upstream_turbine_indices)
    downstream_turbine_indices = [t for t in range(n_turbines) if model_fi.floris.farm.turbine_map.coords[t].x1 > min_downstream_dist]
    n_downstream_turbines = len(downstream_turbine_indices)
        
    # COMPILE TRAINING AND TESTING DATA
    n_inputs = (K_DELAY + 1) * ((2 * len(upstream_turbine_indices)) + 1) # delayed axial induction factor and yaw angle for each upstream turbine, delayed freestream wind speed
    input_labels = generate_input_labels(upstream_turbine_indices)
        
    if COLLECT_DATA:
        pool = mp.Pool(mp.cpu_count())
        wake_field_dfs = pool.starmap(read_datasets, [(p, paths) for p in range(len(paths))])
        pool.close()
        
        ## INSPECT DATA
        if PLOT_DATA:
            learning_turbine_index = downstream_turbine_indices[0]
            upstream_turbine_index = upstream_turbine_indices[0]
            plotting_datasets_idx = [0]
            n_datasets = len(wake_field_dfs)
            fig, axs = plt.subplots(6, 1, sharex=True)
            for df_idx in plotting_datasets_idx:
                df = wake_field_dfs[df_idx]
                for field in df.columns[1:]:
                    for field_type_idx, field_type in enumerate([f'AxIndFactors_{upstream_turbine_index}', f'YawAngles_{upstream_turbine_index}',
                                                                 'FreestreamWindSpeed', 'FreestreamWindDir', 
                                                                 f'TurbineWindSpeeds_{learning_turbine_index}', f'TurbineWindDirs_{learning_turbine_index}']):
                        if field_type in field:
                            row_idx = field_type_idx
                        
                            axs[row_idx].plot(df['Time'], df[field])
                            axs[row_idx].set(title=field)
                    
                    axs[-1].set(xlabel='Time [s]')
            plt.subplots_adjust(wspace=0.6, hspace=0.4)
            plt.show()
        
        # t = wake_field_dfs[0]['Time'].to_numpy()

        X = defaultdict(list)
        y = defaultdict(list)

        pool = mp.Pool(mp.cpu_count())
        res = pool.starmap(split_data, [(model_fi, case_df, upstream_turbine_indices, downstream_turbine_indices) for case_df in wake_field_dfs])
        pool.close()
                
        for X_train, y_train, X_test, y_test, X_full, y_full in res:
            X['train'].append(X_train)
            y['train'].append(y_train)
            X['test'].append(X_test)
            y['test'].append(y_test)
            X['full'].append(X_full)
            y['full'].append(y_full)
         
        for dataset_type in ['train', 'test', 'full']:
            X[dataset_type] = np.vstack(X[dataset_type])
            y[dataset_type] = np.vstack(y[dataset_type])

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
    if PLOT_DATA:
        learning_turbine_index = 0
        upstream_turbine_index = upstream_turbine_indices[0]
        
        start_indices = [0]
        n_datapoints = X['full'].shape[0]
        for row_idx in range(n_datapoints - 1):
            if X['full'][row_idx + 1, input_labels.index('Time')] < X['full'][row_idx, input_labels.index('Time')]:
                start_indices.append(row_idx + 1)
        start_indices.append(n_datapoints)
        
        fig_ip, axs_ip = plt.subplots(2, 3, sharex=True)
        # downstream_dist = model_fi.floris.farm.turbine_map.coords[downstream_turbine_indices[learning_turbine_index]].x1
        # freestream_ws = 8 # model_fi.floris.farm.flow_field.mean_wind_speed
        # wake_propagation_speed = downstream_dist / freestream_ws
        delay_indices = [0, int(K_DELAY // 2)]
        
        for ts_idx in range(len(start_indices) - 1):
            start_t_idx = start_indices[ts_idx]
            end_t_idx = start_indices[ts_idx + 1]
            
            for row_idx, delay_idx in enumerate(delay_indices):
                for col_idx, field_type in  enumerate([f'AxIndFactors_{upstream_turbine_index}', f'YawAngles_{upstream_turbine_index}', 'FreestreamWindSpeed']):
                    field = f'{field_type}_minus{delay_idx}'
                    field_idx = input_labels.index(field)
            
                    axs_ip[row_idx, col_idx].plot(X['full'][start_t_idx:end_t_idx, input_labels.index('Time')], X['full'][start_t_idx:end_t_idx, field_idx], label=f'Time-Series {ts_idx}')
                    axs_ip[row_idx, col_idx].set(title=field)
            
                    axs_ip[-1, col_idx].set(xlabel='Time [s]', xticks=X['full'][K_DELAY::int(60//DT), input_labels.index('Time')])
        axs_ip[0, 0].legend()
        
        plt.subplots_adjust(wspace=0.6, hspace=0.4)
        plt.show()
        
        fig_op, axs_op = plt.subplots(1, 1, sharex=True)
        for ts_idx in range(len(start_indices) - 1):
            start_t_idx = start_indices[ts_idx]
            end_t_idx = start_indices[ts_idx + 1]
            axs_op.plot(X['full'][start_t_idx:end_t_idx, input_labels.index('Time')], y['full'][start_t_idx:end_t_idx, learning_turbine_index], label=f'Time-Series {ts_idx}')
            axs_op.set(title=f'Turbine {downstream_turbine_indices[learning_turbine_index]} Wind Speed [m/s]')
            
            axs_op.set(xlabel='Time [s]', xticks=X['full'][K_DELAY::int(60//DT), input_labels.index('Time')])
        axs_op.legend()
        plt.subplots_adjust(wspace=0.6, hspace=0.4)
        plt.show()
    
    if  NORMALIZE_DATA:
        X_norm = {}
        y_norm = {}
        # exclude Time column
        X_scalar = StandardScaler().fit(X['full'][:, 1:])
        y_scalar = StandardScaler().fit(y['full'])
        
        for data_type in ['train', 'test']:
            X_norm[data_type] = X_scalar.transform(X[data_type][:, 1:])
            y_norm[data_type] = y_scalar.transform(y[data_type])
        
        for var in ['X_norm', 'y_norm', 'X_scalar', 'y_scalar']:
            with open(os.path.join(data_dir, var), 'wb') as f:
                pickle.dump(locals()[var], f)
            
    else:
        for var in ['X_norm', 'y_norm', 'X_scalar', 'y_scalar']:
            with open(os.path.join(data_dir, var), 'rb') as f:
                locals()[var] = pickle.load(f)
    
    for t_idx, learning_turbine_index in enumerate(downstream_turbine_indices):
        
        ## PARAMETERIZE GAUSSIAN PROCESS REGRESSOR
        kernel = ConstantKernel(constant_value=2) * RBF(length_scale=100) # DotProduct() + WhiteKernel()
        gpr = GaussianProcessRegressor(kernel=kernel, 
                                       optimizer=optimizer, 
                                       normalize_y=True, n_restarts_optimizer=100, alpha=1e-10, 
                                       random_state=0)
        gpr_fit = gpr.fit(X_norm['train'], y_norm['train'][:, t_idx])
        for dataset_type in ['train', 'test']:
            print(f'Normalized Score on {dataset_type} Data:', gpr_fit.score(X_norm[dataset_type], y_norm[dataset_type][:, t_idx]), sep=' ') # test on training/testing data
            print(f'Unnormalized Score on {dataset_type} Data:', gpr_fit.score(X[dataset_type][:, 1:], y[dataset_type][:, t_idx]), sep=' ') # test on training/testing data
        
        ## PLOT TRAINING/TEST DATA AND GP PREDICTION
        mean = {}
        std = {}
        sort_idx = {}
        # field_indices = [idx for idx, l in enumerate(input_labels) if 'AxIndFactors_0' in l]
        upstream_turbine_index = upstream_turbine_indices[0]
        field_type = f'AxIndFactors_{upstream_turbine_index}'
        fields = [f'{field_type}_minus{i}' for i in [0, int(K_DELAY // 2), K_DELAY]]
        field_indices = [input_labels.index(f) for f in fields]
        fig, ax = plt.subplots(len(field_indices), 2)
        
        for d, dataset_type in enumerate(['train', 'test']):
            mean[dataset_type], std[dataset_type] = gpr_fit.predict(X_norm[dataset_type], return_std=True)
            mean[dataset_type] = y_scalar.inverse_transform(mean[dataset_type][:, np.newaxis]).squeeze()
            std[dataset_type] = y_scalar.inverse_transform(std[dataset_type][:, np.newaxis]).squeeze()
            
            for ax_idx, field_idx in enumerate(field_indices):
                sort_idx[dataset_type] = np.argsort(X[dataset_type][:, field_idx])

                ax[ax_idx, d].plot(X[dataset_type][sort_idx[dataset_type], field_idx], mean[dataset_type][sort_idx[dataset_type]], label='Predicted')
                ax[ax_idx, d].fill_between(X[dataset_type][sort_idx[dataset_type], field_idx], 
                                mean[dataset_type][sort_idx[dataset_type]] - std[dataset_type][sort_idx[dataset_type]], 
                                mean[dataset_type][sort_idx[dataset_type]] + std[dataset_type][sort_idx[dataset_type]], 
                                alpha=0.1, label='Std. Dev')
                
                ax[ax_idx, d].scatter(X[dataset_type][sort_idx[dataset_type], field_idx], y[dataset_type][sort_idx[dataset_type]], 
                                      linestyle='dashed', color='black', label='Measurements')
                
                ax[ax_idx, d].set(title=f'TurbineWindSpeeds_{learning_turbine_index} - {dataset_type}', 
                                  xlabel=f'{input_labels[field_idx]}', 
                                  xlim=[min(X[dataset_type][:, field_idx]), max(X[dataset_type][:, field_idx])])
                
                # ax[ax_idx, d].legend()
        plt.subplots_adjust(wspace=0.6, hspace=0.4)
                
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
            
            downstream_distance = np.array([model_fi.floris.farm.turbine_map.coords[i].x1 - min_downstream_dist for i in downstream_turbine_indices])
            freestream_ws = row['FreestreamWindSpeed']
            delay_dt = (2 * downstream_distance) * (1 / K_DELAY) * (1 / freestream_ws)
            effective_dk = int(delay_dt // DT)
            
            sim_fi.reinitialize_flow_field(wind_speed=row['FreestreamWindSpeed'], wind_direction=row['FreestreamWindDir'], sim_time=sim_time)
            
            upstream_yaw_angles = [row[f'YawAngles_{t}'] for t in upstream_turbine_indices]
            upstream_ax_ind_factors = [row[f'AxIndFactors_{t}'] for t in upstream_turbine_indices]
            
            sim_fi.calculate_wake(sim_time=sim_time, 
                            yaw_angles=upstream_yaw_angles, 
                            axial_induction=upstream_ax_ind_factors)
            
            if MODEL_TYPE == 'error':
                
                model_fi.reinitialize_flow_field(wind_speed=row['FreestreamWindSpeed'], wind_direction=row['FreestreamWindDir'])
                model_fi.calculate_wake(yaw_angles=upstream_yaw_angles, 
                                        axial_induction=upstream_ax_ind_factors)
                # y_modeled = model_fi.floris.farm.wind_map.turbine_wind_speed[learning_turbine_index]
                y_modeled = model_fi.floris.farm.turbines[learning_turbine_index].average_velocity
            
            # if not enough auto-regressive inputs exist to collect, skip to next time-step in data
            if row_idx < (K_DELAY * effective_dk):
                continue    
             
            # y_true.append(sim_fi.floris.farm.wind_map.turbine_wind_speed[learning_turbine_index])
            y_true.append(sim_fi.floris.farm.turbines[learning_turbine_index].average_velocity)
            
            X_ip = X_scalar.transform(generate_input_vector(test_case_df, row_idx, upstream_turbine_indices, effective_dk)[np.newaxis, 1:])
            mean, std = gpr_fit.predict(X_ip, return_std=True)
            
            if MODEL_TYPE == 'value':
                y_pred.append(y_scalar.inverse_transform(mean[np.newaxis, :]))
            elif MODEL_TYPE == 'error':
                y_pred.append(y_modeled + y_scalar.inverse_transform(mean[np.newaxis, :]))
                
            y_std.append(y_scalar.inverse_transform(std[np.newaxis, :]))
        
        y_pred = np.concatenate(y_pred).squeeze()
        y_std = np.concatenate(y_std).squeeze()
        
        fig, ax = plt.subplots(1, 1)
        ax.set(title=f'Turbine {learning_turbine_index} Wind Speed [m/s]', xlabel='Time [s]')
        ax.plot(time[K_DELAY * effective_dk:], y_true, label='True')
        ax.plot(time[K_DELAY * effective_dk:], y_pred, label='Predicted Mean')
        ax.fill_between(time[K_DELAY * effective_dk:], y_pred - y_std, y_pred + y_std, alpha=0.1, label='Predicted Std. Dev.')
        ax.legend()
        plt.show()
        plt.savefig(os.path.join(fig_dir, 'true_vs_predicted_sim.png'))