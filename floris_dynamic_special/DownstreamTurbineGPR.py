from threading import currentThread
from tkinter.messagebox import NO
from floridyn_special.tools.floris_interface import FlorisInterface as DynFlorisInterface
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import pickle
from scipy.optimize import minimize
from plotting import plot_prediction_vs_input, plot_prediction_vs_time, plot_measurements_vs_time, plot_raw_measurements_vs_time
from preprocessing import get_paths, read_datasets, collect_raw_data, generate_input_labels, generate_input_vector, split_all_data
from numpy.linalg import norm
import sys
import ctypes

# GOAL: learn propagation speed
# GOAL: changers in ax_ind factor of upstream turbines -> changes in effective wind speed at downstream turbines
# Plug sequence ax ind factor into steady-state model -> velocity deficit at downstream turbines, find time span between ax_ind factor 0 and expected velocity deficit
# Find which ax_ind factors correlates most with effective wind speed, find how std (low covariance) varies with different input dimensions
# Start with constant wind speed and dir
# K_DELAY could vary depending on wind speed; workaround: normalized dt by wind speed, @ 10m/s higher sampling rate than 5m/s => constant number of input delays; divide time index by wind speed e.g. wave number [1/m]

GP_CONSTANTS = {'PROPORTION_TRAINING_DATA': 0.85,
                'DT': 1,
                'K_DELAY': 2, # unconservative estimate: (distance to upstream wind turbines / freestream wind speed) / dt, consider half of freestream wind speed for conservative estimate
                'MODEL_TYPE': 'value', # 'error'
                'N_CASES': -1,
                'COLLECT_RAW_DATA': True,
                'NORMALIZE_DATA': False,
                'MODEL_GP': True,
                'PLOT_DATA': True,
                'SIMULATE_DATASET_INDICES': [0],
                'MAX_TRAINING_SIZE': 100,
                'UPSTREAM_RADIUS': 1000,
                'BATCH_SIZE': 10
}

if sys.platform == 'darwin':
    FARM_LAYOUT = '2turb'
    SAVE_DIR = f'./{FARM_LAYOUT}_wake_field_cases'
    FLORIS_DIR = f'./{FARM_LAYOUT}_floris_input.json'
    DATA_DIR = './data'
    FIG_DIR = './figs'
elif sys.platform == 'linux':
    FARM_LAYOUT = '9turb'
    SAVE_DIR = f'/scratch/ahenry/{FARM_LAYOUT}_wake_field_cases'
    FLORIS_DIR = f'./{FARM_LAYOUT}_floris_input.json'
    DATA_DIR = '/scratch/ahenry/data'
    FIG_DIR = '/scratch/ahenry/figs'

# TODO
# 1) generate datasets with layout as in Jean's SOWFA data, test and plot on mac for 2 turb case, run and save on eagle for 9 turb case
# 2) add noise to training data measurements
# 3) formulate and implement exploration maximization algorithm
# 4) implement Matern kernel

# 5) Simulate GP vs true predictions for all datasets

class DownstreamTurbineGPR:
    def __init__(self, kernel, optimizer, X_scalar, y_scalar, max_training_size, n_inputs, n_outputs, turbine_index, upstream_turbine_indices, model_type):
        self.n_inputs = n_inputs
        self.n_outputs = n_outputs
        self.X_train = np.zeros((0, n_inputs))
        self.y_train = np.zeros((0, n_outputs))
        self.X_scalar = X_scalar
        self.y_scalar = y_scalar
        self.max_training_size = max_training_size
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=optimizer, 
            normalize_y=True, n_restarts_optimizer=10, alpha=1e-10, 
            random_state=0)
        self.turbine_index = turbine_index
        self.upstream_turbine_indices = upstream_turbine_indices
        self.model_type = model_type
        self.k_added = []

    def compute_effective_dk(self, system_fi, current_measurement_rows, k_delay=GP_CONSTANTS['K_DELAY'], dt=GP_CONSTANTS['DT']):
        downstream_distance = system_fi.floris.farm.turbine_map.coords[self.turbine_index].x1 - system_fi.min_downstream_dist
        
        freestream_wind_speed = current_measurement_rows['FreestreamWindSpeed']
        delay_dt = (2 * downstream_distance) * (1 / k_delay) * (1 / freestream_wind_speed)
        effective_dk = (delay_dt // dt).astype(int) 
        
        return effective_dk
    
    def add_data(self, measurements_df, system_fi, k_delay=GP_CONSTANTS['K_DELAY'], dt=GP_CONSTANTS['DT']):
        
        new_current_rows = measurements_df.loc[(max(self.k_added) if len(self.k_added) else -1) + 1:]
        effective_dk = self.compute_effective_dk(system_fi, new_current_rows, k_delay=k_delay, dt=dt)
        
        # for each current time-step, that has not been added, which has enought historic inputs behind it, in measurements_df, add the new training inputs
        k_to_add, = np.where(new_current_rows.index >= (k_delay * effective_dk))
        
        if len(k_to_add) == 0:
            return
        
        X = self.X_scalar.transform([generate_input_vector(measurements_df, k, self.upstream_turbine_indices, effective_dk.loc[k], k_delay)[1] for k in k_to_add])
        y = self.y_scalar.transform(measurements_df.loc[k_to_add, f'TurbineWindSpeeds_{self.turbine_index}'].to_numpy().reshape(-1, self.n_outputs))
        
        self.k_added = self.k_added + list(k_to_add) 
         
        # flag this time-step of training data as being added
        # measurements_df.loc[k_to_add, 'IsAdded'] = True
        
        # drop the historic inputs we no longer need for ANY downstream turbine
        # measurements_df.drop(measurements_df.loc[measurements_df['IsAdded']], inplace=True)
        
        self._add_data(X, y)
         
    def _add_data(self, new_X_train, new_y_train):
        self.X_train = np.vstack([self.X_train, new_X_train])[-self.max_training_size:, :]
        self.y_train = np.vstack([self.y_train, new_y_train])[-self.max_training_size:, :]
    
    def fit(self):
        return self.gpr.fit(self.X_train, self.y_train)
    
    def save(self):
        with open(os.path.join(DATA_DIR, f'gpr_{self.turbine_index}'), 'wb') as f:
            pickle.dump(self, f)
    
    def predict(self, measurements_df, system_fi, model_fi=None, k_delay=GP_CONSTANTS['K_DELAY'], dt=GP_CONSTANTS['DT']):
        
        time_step_data = measurements_df.iloc[-1]
        k = measurements_df.index[-1]
        
        # system_fi.reinitialize_flow_field(wind_speed=time_step_data['FreestreamWindSpeed'], wind_direction=time_step_data['FreestreamWindDir'], sim_time=sim_time)
        
        upstream_wind_speeds = [time_step_data[f'TurbineWindSpeeds_{t}'] for t in self.upstream_turbine_indices]
        upstream_yaw_angles = [time_step_data[f'YawAngles_{t}'] for t in self.upstream_turbine_indices]
        upstream_ax_ind_factors = [time_step_data[f'AxIndFactors_{t}'] for t in self.upstream_turbine_indices]
        
        if self.model_type == 'error':
                
            model_fi.reinitialize_flow_field(wind_speed=time_step_data['FreestreamWindSpeed'], wind_direction=time_step_data['FreestreamWindDir'])
            model_fi.calculate_wake(yaw_angles=upstream_yaw_angles, 
                                    axial_induction=upstream_ax_ind_factors)
            # y_modeled = model_fi.floris.farm.wind_map.turbine_wind_speed[learning_turbine_index]
            y_modeled = model_fi.floris.farm.turbines[self.turbine_index].average_velocity
            
        effective_dk = self.compute_effective_dk(system_fi, time_step_data, k_delay=k_delay, dt=dt)
        if k >= (k_delay * effective_dk):
        
            y_true = system_fi.floris.farm.turbines[self.turbine_index].average_velocity
            X_ip = self.X_scalar.transform(generate_input_vector(measurements_df, k, self.upstream_turbine_indices, effective_dk, k_delay)[np.newaxis, :])
            mean, std = self.predict(X_ip, return_std=True)
            
            if self.model_type == 'value':
                y_pred = self.y_scalar.inverse_transform(mean[np.newaxis, :])
            elif self.model_type == 'error':
                y_pred = y_modeled + self.y_scalar.inverse_transform(mean[np.newaxis, :])
                
            y_std = self.y_scalar.inverse_transform(std[np.newaxis, :])

            return y_true, y_pred, y_std
        else:
            return np.nan, np.nan, np.nan
     
    def simulate(self, simulation_df, system_fi, model_fi=None, floris_dir=GP_CONSTANTS['FLORIS_DIR'], model_type=GP_CONSTANTS['MODEL_TYPE'], k_delay=GP_CONSTANTS['K_DELAY'], dt=GP_CONSTANTS['K_DELAY']):
        """_summary_
        for given time-series of freestream-wind speed and open-loop axIndFactors and YawAngles, simulate the true vs. GP-predicted effective wind speed at this downstream turbine

        Args:
            simulation_df (_type_): _description_
            system_fi (_type_): _description_
            model_fi (_type_, optional): _description_. Defaults to None.
            floris_dir (_type_, optional): _description_. Defaults to GP_CONSTANTS['FLORIS_DIR'].
            model_type (_type_, optional): _description_. Defaults to GP_CONSTANTS['MODEL_TYPE'].
            k_delay (_type_, optional): _description_. Defaults to GP_CONSTANTS['K_DELAY'].
            dt (_type_, optional): _description_. Defaults to GP_CONSTANTS['K_DELAY'].

        Returns:
            _type_: _description_
        """
        
        y_true = []
        y_pred = []
        y_std = []
        freestream_wind_speeds = simulation_df['FreestreamWindSpeed'].to_numpy()
        ax_ind_factors = np.hstack([simulation_df[f'AxIndFactors_{t_idx}'].to_numpy()[np.newaxis, :] for t_idx in self.upstream_turbine_indices])
        yaw_angles = np.hstack([simulation_df[f'YawAngles_{t_idx}'].to_numpy()[np.newaxis, :] for t_idx in self.upstream_turbine_indices])
        time = simulation_df['Time']
        
        sim_fi = DynFlorisInterface(floris_dir)
        sim_fi.floris.farm.flow_field.mean_wind_speed = simulation_df['FreestreamWindSpeed'].mean()
        
        if model_type == 'error':
            model_fi.floris.farm.flow_field.mean_wind_speed = simulation_df['FreestreamWindSpeed'].mean()
        
        for k, sim_time in enumerate(simulation_df['Time']):
            y_true_k, y_pred_k, y_std_k = self.predict(simulation_df, sim_time, system_fi, model_fi)
            
            if not np.isnan(y_true_k):    
                y_true.append(y_true_k)
                y_pred.append(y_pred_k)
                y_std.append(y_std_k)
        
        y_pred = np.concatenate(y_pred).squeeze()
        y_std = np.concatenate(y_std).squeeze()
        
        X_ts = {'Freestream Wind Speed [m/s]': freestream_wind_speeds,
                'Axial Induction Factors [-]': ax_ind_factors,
                'Yaw Angles [deg]': yaw_angles}
        y_ts = {f'Turbine {self.turbine_index} Wind Speed [m/s]': {
                    'y_true': y_true, 
                    'y_pred': y_pred, 
                    'y_std': y_std
                    }
                }
        
        return time, X_ts, y_ts
    
def optimizer(fun, initial_theta, bounds):
    res = minimize(fun, initial_theta, 
                   method="L-BFGS-B", jac=True, 
                   bounds=bounds, 
                   options={'maxiter': 1000})
    theta_opt = res.x
    func_min = res.fun
    return theta_opt, func_min

# if __name__ == '__main__':
def get_system_info():
    
    ## GET SYSTEM INFORMATION
    system_fi = DynFlorisInterface(GP_CONSTANTS['FLORIS_DIR'])
    n_turbines = len(system_fi.floris.farm.turbines)
    system_fi.max_downstream_dist = max(system_fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
    system_fi.min_downstream_dist = min(system_fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
    # exclude most downstream turbine
    system_fi.upstream_turbine_indices = [t for t in range(n_turbines) if system_fi.floris.farm.turbine_map.coords[t].x1 < system_fi.max_downstream_dist]
    system_fi.downstream_turbine_indices = [t for t in range(n_turbines) if system_fi.floris.farm.turbine_map.coords[t].x1 > system_fi.min_downstream_dist]
     
    return system_fi
 
def get_model_prior():   
    ## DEFINE PRIOR MODEL
    model_fi = DynFlorisInterface(GP_CONSTANTS['FLORIS_DIR'])
    return model_fi

def get_data(system_fi, model_type=GP_CONSTANTS['MODEL_TYPE'], k_delay=GP_CONSTANTS['K_DELAY'], proportion_training_data=GP_CONSTANTS['PROPORTION_TRAINING_DATA'], dt=GP_CONSTANTS['DT'], model_fi=None, collect_raw_data_bool=GP_CONSTANTS['COLLECT_RAW_DATA'], plot_data_bool=GP_CONSTANTS['PLOT_DATA']):
    ## FETCH RAW DATA
    for dir in [DATA_DIR, FIG_DIR]:
        if not os.path.exists(dir):
            os.mkdir(dir)
    
    csv_paths = get_paths(SAVE_DIR, n_cases=GP_CONSTANTS['N_CASES'])
    plotting_ds_turbine_index = system_fi.downstream_turbine_indices[0]
    plotting_us_turbine_index = system_fi.upstream_turbine_indices[0]
    
    # + ['FreestreamWindSpeed'] \
    current_input_labels = ['Time'] + ['FreestreamWindDir'] \
    + [f'TurbineWindSpeeds_{t}' for t in system_fi.upstream_turbine_indices] \
        + [f'AxIndFactors_{t}' for t in system_fi.upstream_turbine_indices] \
        + [f'YawAngles_{t}' for t in system_fi.upstream_turbine_indices] \
        + [f'TurbineWindSpeeds_{t}' for t in system_fi.downstream_turbine_indices]
        
    if collect_raw_data_bool:
        wake_field_dfs = collect_raw_data(SAVE_DIR, csv_paths)
        
        ## INSPECT DATA
        if plot_data_bool:
            plotting_df_idx = [0, 1]
            plotting_dfs = [wake_field_dfs[df_idx] for df_idx in plotting_df_idx]
            # 'FreestreamWindSpeed', 
            input_types = ['FreestreamWindDir', 
                           f'TurbineWindSpeeds_{plotting_us_turbine_index}',
                           f'AxIndFactors_{plotting_us_turbine_index}', 
                           f'YawAngles_{plotting_us_turbine_index}',
                           f'TurbineWindSpeeds_{plotting_ds_turbine_index}', 
                           f'TurbineWindDirs_{plotting_ds_turbine_index}']
            
            _, ax = plt.subplots(len(input_types), 1, sharex=True)
            plot_raw_measurements_vs_time(ax, plotting_dfs, input_types) 
            plt.show()

        time, X, y, measurements_dfs = split_all_data(model_fi, system_fi, wake_field_dfs, current_input_labels, proportion_training_data, model_type, k_delay, dt)

        for var in ['time', 'X', 'y', 'measurements_dfs']:
            with open(os.path.join(DATA_DIR, var), 'wb') as f:
                pickle.dump(locals()[var], f)
        # with open(os.path.join(DATA_DIR,  'measurements_dfs'), 'wb') as f:
        #         pickle.dump('measurements_dfs', f)
            
    else:
        mmry_dict = {}
        for var in ['time', 'X', 'y', 'measurements_dfs']:
            with open(os.path.join(DATA_DIR, var), 'rb') as f:
                # exec("nonlocal " + var)
                # sys._getframe(1).f_locals[var] = pickle.load(f)
                # ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(sys._getframe(1)), ctypes.c_int(0))
                # exec(var + " = pickle.load(f)")
                mmry_dict[var] = pickle.load(f)

            # locals()[var] = mmry_dict[var]
        time = mmry_dict['time']
        X = mmry_dict['X']
        y = mmry_dict['y']
        measurements_dfs = mmry_dict['measurements_dfs']

    input_labels = generate_input_labels(system_fi.upstream_turbine_indices, k_delay)
    

    
    ## INSPECT DATA
    if plot_data_bool:
        # fig = plt.figure(constrained_layout=True)
        
        # gs = GridSpec(3, 3, figure=fig)
        delay_indices = [0, int(GP_CONSTANTS['K_DELAY'] // 2)]
        input_types = [f'AxIndFactors_{plotting_us_turbine_index}', f'YawAngles_{plotting_us_turbine_index}', 'FreestreamWindSpeed']
        output_labels = [f'Turbine {plotting_ds_turbine_index} Wind Speed [m/s]']
        fig, axs = plt.subplots(len(delay_indices) + 1 + int(len(output_labels) // len(input_types)), len(input_types), sharex=True)
        plot_measurements_vs_time(axs, time['full'], X['full'], y['full'], input_labels, input_types, delay_indices, output_labels)
        plt.show()
        
    return time, X, y, current_input_labels, input_labels, measurements_dfs

def init_gprs(system_fi, X_scalar, y_scalar, input_labels, max_training_size=GP_CONSTANTS['MAX_TRAINING_SIZE']):
    ## GENERATE GP MODELS FOR EACH DOWNSTREAM TURBINE'S WIND SPEED
    gprs = []
    for ds_t_idx in system_fi.downstream_turbine_indices:
        # include all turbines with upstream radius of this one TODO check 1 row for 9turb case
        upstream_turbine_indices = [us_t_idx for us_t_idx in system_fi.upstream_turbine_indices 
                                    if norm([system_fi.floris.farm.turbine_map.coords[us_t_idx].x1 - system_fi.floris.farm.turbine_map.coords[ds_t_idx].x1,
                                             system_fi.floris.farm.turbine_map.coords[us_t_idx].x3 - system_fi.floris.farm.turbine_map.coords[ds_t_idx].x3], ord=2)
                                    <= GP_CONSTANTS['UPSTREAM_RADIUS']]
         
        # PARAMETERIZE GAUSSIAN PROCESS REGRESSOR
        kernel = ConstantKernel(constant_value=5) * RBF(length_scale=1) # DotProduct() + WhiteKernel()
        gpr = DownstreamTurbineGPR(kernel, optimizer, X_scalar, y_scalar, 
                                   max_training_size=max_training_size, 
                                   n_inputs=len(input_labels), n_outputs=1, 
                                   turbine_index=ds_t_idx,
                                   upstream_turbine_indices=upstream_turbine_indices,
                                   model_type=GP_CONSTANTS['MODEL_TYPE'])
        
        gprs.append(gpr)
    return gprs

def evaluate_gprs(gprs, system_fi, X_norm, y_norm, input_labels, plot_data=GP_CONSTANTS['PLOT_DATA'], k_delay=GP_CONSTANTS['K_DELAY']):
    for t_idx, gpr in enumerate(gprs):
        for dataset_type in ['train', 'test']:
            print(f'Score on {dataset_type} Data:', 
                   gpr.score(X_norm[dataset_type], y_norm[dataset_type][:, t_idx]), sep=' ') # test on training/testing data
                                                                                                                                                                                                                                                                                                                                   
        # PLOT TRAINING/TEST DATA AND GP PREDICTION vs AX IND FACTOR INPUT
        if plot_data:
            plotting_us_turbine_index = system_fi.upstream_turbine_indices[0]   
            input_type = f'AxIndFactors_{plotting_us_turbine_index}'
            inputs = [f'{input_type}_minus{i}' for i in [0, int(k_delay // 2), k_delay]]
            _, ax = plt.subplots(len(inputs), 2)
            plot_prediction_vs_input(ax[:, 0], gpr, inputs, input_labels, X_norm['train'], y_norm['train'], gpr.X_scalar, gpr.y_scalar, plotting_us_turbine_index, 'train')
            plot_prediction_vs_input(ax[:, 1], gpr, inputs, input_labels, X_norm['test'], y_norm['test'], gpr.X_scalar, gpr.y_scalar, plotting_us_turbine_index, 'test')
            plt.show()
            plt.savefig(os.path.join(FIG_DIR, 'training_test_prediction.png'))
    
def refit_gprs(gprs, X_norm, y_norm):
    for t_idx, gpr in enumerate(gprs):
        gpr.add_data(X_norm['train'], y_norm['train'][:, t_idx])
        gpr.fit()
    return gprs

def test_gpr(gpr, system_fi, simulation_case_idx, csv_paths=None, wake_field_dfs=None, model_fi=None, 
             plot_data=GP_CONSTANTS['PLOT_DATA'], collect_raw_data=GP_CONSTANTS['COLLECT_RAW_DATA'], floris_dir=GP_CONSTANTS['FLORIS_DIR'], 
             model_type=GP_CONSTANTS['MODEL_TYPE'], k_delay=GP_CONSTANTS['K_DELAY'], dt=GP_CONSTANTS['K_DELAY']):        
    ## MODEL WAKE-FIELD DATA AT TURBINES AND COMPARE TO GP PREDICTIONS
    if collect_raw_data:
        simulation_df = wake_field_dfs[simulation_case_idx]
    else:
        simulation_df = read_datasets(simulation_case_idx, csv_paths)
        
    time, X_ts, y_ts = gpr.simulate(simulation_df, system_fi, model_fi=model_fi, floris_dir=floris_dir, model_type=model_type, k_delay=k_delay, dt=dt)
    
    if plot_data:
        _, ax = plt.subplots(4, 1, sharex=True)
        plot_prediction_vs_time(ax, time, X_ts, y_ts)
        plt.show()
        plt.savefig(os.path.join(FIG_DIR, 'true_vs_predicted_sim.png'))