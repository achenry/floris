from floridyn.tools.floris_interface import FlorisInterface as DynFlorisInterface
from sklearn.gaussian_process import GaussianProcessRegressor

import matplotlib.pyplot as plt
import numpy as np
import os
import pickle
from scipy.optimize import minimize
from plotting import plot_prediction_vs_input, plot_prediction_vs_time, plot_measurements_vs_time, plot_raw_measurements_vs_time
from preprocessing import get_paths, read_datasets, collect_raw_data, generate_input_labels, generate_input_vector, split_all_data
from numpy.linalg import norm
import sys

# GOAL: learn propagation speed
# GOAL: changers in ax_ind factor of upstream turbines -> changes in effective wind speed at downstream turbines
# Plug sequence ax ind factor into steady-state model -> velocity deficit at downstream turbines, find time span between ax_ind factor 0 and expected velocity deficit
# Find which ax_ind factors correlates most with effective wind speed, find how std (low covariance) varies with different input dimensions
# Start with constant wind speed and dir
# K_DELAY could vary depending on wind speed; workaround: normalized dt by wind speed, @ 10m/s higher sampling rate than 5m/s => constant number of input delays; divide time index by wind speed e.g. wave number [1/m]

GP_CONSTANTS = {'PROPORTION_TRAINING_DATA': 0.8,
                'DT': 1,
                'K_DELAY': 2, # unconservative estimate: (distance to upstream wind turbines / freestream wind speed) / dt, consider half of freestream wind speed for conservative estimate
                'MODEL_TYPE': 'error', # 'value, error'
                'N_CASES': -1,
                'COLLECT_RAW_DATA': True,
                'MODEL_GP': True,
                'PLOT_DATA': False,
                'SIMULATE_DATASET_INDICES': [0],
                'MAX_TRAINING_SIZE': 500,
                'UPSTREAM_RADIUS': 1000,
                'BATCH_SIZE': 10,
                'STD_THRESHOLD': 0.1
}

N_TEST_POINTS_PER_COORD = 2
AX_IND_FACTOR_TEST_POINTS = np.linspace(0.11, 0.33, N_TEST_POINTS_PER_COORD)
YAW_ANGLE_TEST_POINTS = np.linspace(-15, 15, N_TEST_POINTS_PER_COORD)
UPSTREAM_WIND_SPEED_TEST_POINTS = np.linspace(8, 12, N_TEST_POINTS_PER_COORD)
UPSTREAM_WIND_DIR_TEST_POINTS = np.linspace(250, 270, N_TEST_POINTS_PER_COORD)

if sys.platform == 'darwin':
    FARM_LAYOUT = '2turb'
    SAVE_DIR = f'./{FARM_LAYOUT}_wake_field_cases'
    FLORIS_DIR = f'./{FARM_LAYOUT}_floris_input.json'
    BASE_MODEL_FLORIS_DIR = f'./{FARM_LAYOUT}_base_model_floris_input.json'
    DATA_DIR = './data'
    FIG_DIR = './figs'
elif sys.platform == 'linux':
    FARM_LAYOUT = '9turb'
    SAVE_DIR = f'/scratch/alpine/aohe7145/{FARM_LAYOUT}_wake_field_cases'
    FLORIS_DIR = f'./{FARM_LAYOUT}_floris_input.json'
    BASE_MODEL_FLORIS_DIR = f'./{FARM_LAYOUT}_base_model_floris_input.json'
    DATA_DIR = '/scratch/alpine/aohe7145/wake_gp/data'
    FIG_DIR = '/scratch/alpine/aohe7145/wake_gp/figs'

# TODO
# 1) generate datasets with layout as in Jean's SOWFA data, test and plot on mac for 2 turb case, run and save on eagle for 9 turb case
# 3) formulate and implement exploration maximization algorithm

class DownstreamTurbineGPR:
    def __init__(self, kernel, optimizer, X_scalar, y_scalar,
                 max_training_size, input_labels, n_outputs, turbine_index,
                 upstream_turbine_indices, model_type):
        self.input_labels = input_labels
        self.n_inputs = len(self.input_labels)
        self.n_outputs = n_outputs
        self.X_train = np.zeros((0, self.n_inputs))
        self.y_train = np.zeros((0, self.n_outputs))
        self.k_train = []

        self.X_train_replay = np.zeros((0, self.n_inputs))
        self.y_train_replay = np.zeros((0, self.n_outputs))
        self.k_train_replay = []

        self.X_scalar = X_scalar
        self.y_scalar = y_scalar
        self.max_training_size = max_training_size
        self.max_replay_size = max_training_size
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=optimizer, 
            normalize_y=True, n_restarts_optimizer=10, alpha=1e-10, 
            random_state=0)
        self.turbine_index = turbine_index
        self.upstream_turbine_indices = upstream_turbine_indices
        self.model_type = model_type
        self.X_test = None
        # self.k_added = []

    def compute_test_std(self):
        _, test_std = self.gpr.predict(self.X_test, return_std=True)
        test_std = self.y_scalar.inverse_transform(test_std)
        return test_std / self.X_test.shape[0]

    def compute_effective_dk(self, system_fi, current_measurement_rows, k_delay=GP_CONSTANTS['K_DELAY'],
                             dt=GP_CONSTANTS['DT']):
        # compute maximum number of delayed time-steps to consider in inputs to predict effective wind speed at this wind turbine

        # compute maximum downstream distance of this turbine from influential upstream turbines
        max_downstream_distance = system_fi.floris.farm.turbine_map.coords[self.turbine_index].x1 \
                              - min(system_fi.floris.farm.turbine_map.coords[us_idx].x1
                                    for us_idx in self.upstream_turbine_indices)

        freestream_wind_speed = current_measurement_rows['FreestreamWindSpeed']

        # compute time for wake to propagate from front row to this turbine according to Taylor's frozen wake hypothesis
        base_delay_dt = max_downstream_distance / freestream_wind_speed

        # compute maximum time delay, which will almost certainly include historic upstream changes which influence the current effective wind speed at this turbine
        # consider 2 * the wake propagation time according to Taylor's frozen wake hypothesis
        max_delay_dt = 2 * base_delay_dt

        # divide this delayed time-span by the number of autoregressive inputs
        delay_dt = max_delay_dt / k_delay

        # convert from time to time-steps
        delay_dk = (delay_dt // dt).astype(int)
        
        return delay_dk

    def check_history(self, measurements_df, system_fi, k_delay=GP_CONSTANTS['K_DELAY'], dt=GP_CONSTANTS['DT']):
        # find the time indices in the current measurements dataframe with enough historic inputs to form an autoregressive input
        new_current_rows = measurements_df
        effective_dk = self.compute_effective_dk(system_fi, new_current_rows, k_delay=k_delay, dt=dt)

        # for each current time-step, that has not been added, which has enough historic inputs behind it,
        # in measurements_df, add the new training inputs
        k_idx_to_add = new_current_rows['Time'] >= (k_delay * effective_dk)
        k_to_add = (new_current_rows.loc[k_idx_to_add, 'Time'] // dt).astype(int)
        reduced_effective_dk = effective_dk.loc[k_idx_to_add]
        history_exists = len(k_to_add) > 0

        return k_to_add, effective_dk, reduced_effective_dk, history_exists

    def prepare_data(self, measurements_df, k_to_add, effective_dk, k_delay=GP_CONSTANTS['K_DELAY'], y_modeled=None):
        X = self.X_scalar.transform(
            np.vstack([generate_input_vector(
                measurements_df, k_idx, self.upstream_turbine_indices, effective_dk.loc[k_idx], k_delay)[1]
             for k_idx in k_to_add.index]))

        y_err = measurements_df.loc[k_to_add, f'TurbineWindSpeeds_{self.turbine_index}'].to_numpy() - y_modeled
        y = self.y_scalar.transform(y_err.reshape(-1, self.n_outputs))

        return X, y

    def add_data(self, new_X_train, new_y_train, new_k_train, is_online):

        self.X_train = np.vstack([self.X_train, new_X_train])[-self.max_training_size if self.max_training_size > -1
                                                              else 0:, :]
        self.y_train = np.vstack([self.y_train, new_y_train])[-self.max_training_size if self.max_training_size > -1
                                                              else 0:, :]
        assert self.X_train.shape[0] == self.max_training_size

        if is_online:
            self.k_train = (self.k_train + list(new_k_train))[-self.max_training_size if self.max_training_size > -1
                                                              else 0:]
        else:
            self.k_train = [-1] * self.X_train.shape[0]
    
    def fit(self):
        return self.gpr.fit(self.X_train, self.y_train)
    
    def save(self):
        with open(os.path.join(DATA_DIR, f'gpr_{self.turbine_index}'), 'wb') as f:
            pickle.dump(self, f)
    
    def predict(self, measurements_df, system_fi, y_modeled=None,
                k_delay=GP_CONSTANTS['K_DELAY'], dt=GP_CONSTANTS['DT']):
        
        time_step_data = measurements_df.iloc[-1]
        k = int(time_step_data['Time'] // dt) # measurements_df.index[-1]
        
        # upstream_yaw_angles = [time_step_data[f'YawAngles_{t}'] for t in self.upstream_turbine_indices]
        # upstream_ax_ind_factors = [time_step_data[f'AxIndFactors_{t}'] for t in self.upstream_turbine_indices]

        effective_dk = self.compute_effective_dk(system_fi, time_step_data, k_delay=k_delay, dt=dt)
        if k >= (k_delay * effective_dk):

            X_ip = self.X_scalar.transform(
                [generate_input_vector(
                    measurements_df, k, self.upstream_turbine_indices, effective_dk, k_delay)[1]])
            mean, std = self.gpr.predict(X_ip, return_std=True)

            y_pred = y_modeled + self.y_scalar.inverse_transform(mean).squeeze()
            y_std = self.y_scalar.inverse_transform([std]).squeeze()

            return y_pred, y_std
        else:
            return np.nan, np.nan

    def choose_new_data(self, X_train_potential, y_train_potential, k_train_potential,
                        n_datapoints=GP_CONSTANTS['BATCH_SIZE']):
        """

        Args:
            X_train_new: potential new training data points

        Returns:

        """
        # OPTION A:
        # select MAX_TRAINING_SIZE datapoints from self.X_train_all at random

        # OPTION B:
        # drop BATCH_SIZE datapoints from self.X_train at random and add BATCH_SIZE points with highest predicted variance from new available X_train_new
        assert self.X_train.shape[0] == self.max_training_size

        keep_idx = list(range(self.X_train.shape[0]))
        np.random.shuffle(keep_idx)
        drop_idx = []
        for _ in range(n_datapoints):
            drop_idx.append(keep_idx.pop())
        # drop_idx = np.floor(np.random.uniform(0, 1, n_datapoints) * self.X_train.shape[0]).astype(int)
        # keep_idx = [i for i in range(self.X_train.shape[0]) if i not in drop_idx]
        assert len(drop_idx) + len(keep_idx)

        dropped_X_train = self.X_train[drop_idx, :]
        dropped_y_train = self.y_train[drop_idx, :]
        dropped_k_train = [self.k_train[i] for i in drop_idx]

        self.X_train = self.X_train[keep_idx, :]
        self.y_train = self.y_train[keep_idx, :]
        self.k_train = [self.k_train[i] for i in keep_idx]
        self.fit()

        self.X_train_replay = np.vstack([self.X_train_replay, X_train_potential, dropped_X_train])
        self.y_train_replay = np.vstack([self.y_train_replay, y_train_potential, dropped_y_train])
        self.k_train_replay = self.k_train_replay + list(k_train_potential) + dropped_k_train

        # predict variance for each of new candidate datapoints
        new_dps = []
        for i, k in enumerate(self.k_train_replay):
            X = self.X_train_replay[i, :]
            y = self.y_train_replay[i]
            _, std = self.gpr.predict([X], return_std=True)
            new_dps.append((X, y, std[0], k))

        assert len(new_dps) >= n_datapoints

        # order from highest to lowest predicted standard deviation
        new_std_train = [tup[2] for tup in new_dps]
        p_choice = new_std_train / sum(new_std_train)

        idx_choice = np.random.choice(list(range(len(new_dps))), n_datapoints, replace=False, p=p_choice)

        # remove from replay data that has been selected
        self.X_train_replay = self.X_train_replay[~idx_choice]
        self.y_train_replay = self.y_train_replay[~idx_choice]
        self.k_train_replay = [k for i, k in enumerate(self.k_train_replay) if i not in idx_choice]

        # add to training data
        # new_Xy_train = sorted(new_dps, key=lambda tup: tup[2], reverse=True)[:n_datapoints]
        new_Xy_train = [new_dps[i] for i in idx_choice]
        new_X_train = np.vstack([tup[0] for tup in new_Xy_train])
        new_y_train = np.vstack([tup[1] for tup in new_Xy_train])
        new_k_train = [tup[3] for tup in new_Xy_train]

        assert new_X_train.shape[0] == n_datapoints

        self.add_data(new_X_train, new_y_train, new_k_train, is_online=True)

        assert self.X_train.shape[0] == self.max_training_size

        # truncate the replay buffer
        shuffle_idx = list(range(self.X_train_replay.shape[0]))
        np.random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[:self.max_replay_size]
        self.X_train_replay = self.X_train_replay[shuffle_idx, :]
        self.y_train_replay = self.y_train_replay[shuffle_idx, :]
        self.k_train_replay = [self.k_train_replay[i] for i in shuffle_idx]

        assert self.X_train.shape[0] == self.max_training_size

    def get_domain_window(self, x, window_size=0.1, n_datapoints=10):
        x_window = []
        X_train_width = np.linalg.norm(np.max(self.X_train, axis=0) - np.min(self.X_train, axis=0))
        window_size = X_train_width / self.X_train.shape[0]
        # for each data point to add to window
        for i in range(n_datapoints):
            # choose a random vector direction and length (<= window size)
            dir = np.random.uniform(0, 2*np.pi, self.n_inputs)
            mag = np.random.uniform(0, window_size, 1)
            # add direction cosines
            x_dash = x + mag * np.cos(dir)
            x_window.append(x_dash)

        return x_window

    def simulate(self, simulation_df, system_fi, model_fi=None, floris_dir=FLORIS_DIR,
                 model_type=GP_CONSTANTS['MODEL_TYPE']):
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
        
        for k, sim_time in enumerate(simulation_df['Time']):

            sim_fi.floris.farm.flow_field.mean_wind_speed = simulation_df.loc[k, 'FreestreamWindSpeed']

            if model_type == 'error':
                model_fi.floris.farm.flow_field.mean_wind_speed = simulation_df.loc[k, 'FreestreamWindSpeed']

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
    system_fi = DynFlorisInterface(FLORIS_DIR)
    n_turbines = len(system_fi.floris.farm.turbines)
    system_fi.max_downstream_dist = max(system_fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
    system_fi.min_downstream_dist = min(system_fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
    # exclude most downstream turbine
    system_fi.turbine_indices = list(range(n_turbines))
    system_fi.upstream_turbine_indices = [t for t in range(n_turbines)
                                          if system_fi.floris.farm.turbine_map.coords[t].x1
                                          < system_fi.max_downstream_dist]
    system_fi.downstream_turbine_indices = [t for t in range(n_turbines)
                                            if system_fi.floris.farm.turbine_map.coords[t].x1
                                            > system_fi.min_downstream_dist]
     
    return system_fi
 
def get_base_model():

    ## DEFINE PRIOR MODEL
    model_fi = DynFlorisInterface(BASE_MODEL_FLORIS_DIR)

    n_turbines = len(model_fi.floris.farm.turbines)
    model_fi.max_downstream_dist = max(model_fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
    model_fi.min_downstream_dist = min(model_fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
    # exclude most downstream turbine
    model_fi.turbine_indices = list(range(n_turbines))
    model_fi.upstream_turbine_indices = [t for t in range(n_turbines)
                                         if model_fi.floris.farm.turbine_map.coords[t].x1
                                         < model_fi.max_downstream_dist]
    model_fi.downstream_turbine_indices = [t for t in range(n_turbines)
                                           if model_fi.floris.farm.turbine_map.coords[t].x1
                                           > model_fi.min_downstream_dist]

    return model_fi

def get_dfs(df_indices, proportion_training_data=GP_CONSTANTS['PROPORTION_TRAINING_DATA']):
    ## FETCH RAW DATA
    for dir in [DATA_DIR, FIG_DIR]:
        if not os.path.exists(dir):
            os.mkdir(dir)

    csv_paths = get_paths(SAVE_DIR, df_indices=df_indices)

    wake_field_dfs = collect_raw_data(SAVE_DIR, csv_paths)
    n_training_datasets = int(np.floor(len(wake_field_dfs) * proportion_training_data))
    full_idx = np.arange(len(wake_field_dfs))
    np.random.shuffle(full_idx)
    training_idx = full_idx[:n_training_datasets]
    testing_idx = full_idx[n_training_datasets:]
    wake_field_dfs = {'train': [wake_field_dfs[i] for i in training_idx],
                      'test': [wake_field_dfs[i] for i in testing_idx]}
    return wake_field_dfs

def get_data(measurements_dfs, system_fi, model_type=GP_CONSTANTS['MODEL_TYPE'], k_delay=GP_CONSTANTS['K_DELAY'],
             dt=GP_CONSTANTS['DT'], model_fi=None,
             collect_raw_data_bool=GP_CONSTANTS['COLLECT_RAW_DATA'], plot_data_bool=GP_CONSTANTS['PLOT_DATA']):

    current_input_labels = ['Time'] + ['FreestreamWindSpeed'] + ['FreestreamWindDir'] \
                           + [f'TurbineWindSpeeds_{t}' for t in system_fi.turbine_indices] \
                           + [f'AxIndFactors_{t}' for t in system_fi.turbine_indices] \
                           + [f'YawAngles_{t}' for t in system_fi.turbine_indices]

    if collect_raw_data_bool:

        ## INSPECT DATA
        if plot_data_bool:
            plotting_df_idx = [0]
            plotting_dfs = [measurements_dfs['train'][df_idx] for df_idx in plotting_df_idx]
            plotting_ds_turbine_index = system_fi.downstream_turbine_indices[0]
            plotting_us_turbine_index = system_fi.upstream_turbine_indices[0]
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

        time, X, y = split_all_data(model_fi, system_fi, measurements_dfs, current_input_labels,
                                                      model_type, k_delay, dt)

        for var in ['time', 'X', 'y']:
            with open(os.path.join(DATA_DIR, var), 'wb') as f:
                pickle.dump(locals()[var], f)
            
    else:
        mmry_dict = {}
        for var in ['time', 'X', 'y']:
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
        input_types = [f'AxIndFactors_{plotting_us_turbine_index}', f'YawAngles_{plotting_us_turbine_index}',
                       f'TurbineWindSpeeds_{plotting_us_turbine_index}']
        output_labels = [f'Turbine {plotting_ds_turbine_index} Wind Speed [m/s]']
        fig, axs = plt.subplots(len(delay_indices) + 1 + int(len(output_labels) // len(input_types)), len(input_types), sharex=True)
        plot_measurements_vs_time(axs, time['train'], X['train'], y['train'], input_labels, input_types, delay_indices, output_labels)
        plt.show()
        
    return time, X, y, current_input_labels, input_labels

def init_gprs(system_fi, X_scalar, y_scalar, input_labels, kernel, k_delay, max_training_size=GP_CONSTANTS['MAX_TRAINING_SIZE']):
    ## GENERATE GP MODELS FOR EACH DOWNSTREAM TURBINE'S WIND SPEED
    gprs = []
    for ds_t_idx in system_fi.downstream_turbine_indices:
        # include all turbines with upstream radius of this one
        upstream_turbine_indices = [us_t_idx for us_t_idx in system_fi.upstream_turbine_indices 
                                    if norm([system_fi.floris.farm.turbine_map.coords[us_t_idx].x1 - system_fi.floris.farm.turbine_map.coords[ds_t_idx].x1,
                                             system_fi.floris.farm.turbine_map.coords[us_t_idx].x2 - system_fi.floris.farm.turbine_map.coords[ds_t_idx].x2
                                             ], ord=2)
                                    <= GP_CONSTANTS['UPSTREAM_RADIUS']]
         
        # PARAMETERIZE GAUSSIAN PROCESS REGRESSOR
         # DotProduct() + WhiteKernel()
        turbine_input_labels = [l for l in input_labels
                                if 'TurbineWindSpeeds' not in l or int(l.split('_')[1]) in upstream_turbine_indices]

        gpr = DownstreamTurbineGPR(kernel, optimizer,
                                   X_scalar, y_scalar,
                                   max_training_size=max_training_size, 
                                   input_labels=turbine_input_labels, n_outputs=1,
                                   turbine_index=ds_t_idx,
                                   upstream_turbine_indices=upstream_turbine_indices,
                                   model_type=GP_CONSTANTS['MODEL_TYPE'])

        if os.path.exists(os.path.join(DATA_DIR, f'X_test_ds-{ds_t_idx}_kdelay-{k_delay}')):
            with open(os.path.join(DATA_DIR, f'X_test_ds-{ds_t_idx}_kdelay-{k_delay}'), 'rb') as f:
                gpr.X_test = pickle.load(f)
        else:
            test_vectors = []
            for l in turbine_input_labels:
                if 'AxIndFactors' in l:
                    test_vectors.append(AX_IND_FACTOR_TEST_POINTS)
                elif 'YawAngles' in l:
                    test_vectors.append(YAW_ANGLE_TEST_POINTS)
                elif 'TurbineWindSpeeds' in l:
                    test_vectors.append(UPSTREAM_WIND_SPEED_TEST_POINTS)
                elif 'FreestreamWindDir' in l:
                    test_vectors.append(UPSTREAM_WIND_DIR_TEST_POINTS)

            gpr.X_test = gpr.X_scalar.transform(np.array(np.meshgrid(*test_vectors, copy=False)).T.reshape(-1, len(turbine_input_labels)))

            with open(os.path.join(DATA_DIR, f'X_test_ds-{ds_t_idx}_kdelay-{k_delay}'), 'wb') as f:
                pickle.dump(gpr.X_test, f)

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
             plot_data=GP_CONSTANTS['PLOT_DATA'], collect_raw_data=GP_CONSTANTS['COLLECT_RAW_DATA'], floris_dir=FLORIS_DIR,
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