from floridyn.tools.floris_interface import FlorisInterface as DynFlorisInterface
from sklearn.gaussian_process import GaussianProcessRegressor
import numpy as np
import os
import pickle
from scipy.optimize import minimize
from preprocessing import get_paths, collect_raw_data, generate_input_labels, generate_input_vector
from numpy.linalg import norm

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
                'UPSTREAM_RADIUS': 1000, # only consider those that surround immediately
                'BATCH_SIZE': 10,
                'STD_THRESHOLD': 0.1,
                'N_TEST_POINTS': 100
}

N_TEST_POINTS_PER_COORD = 3
AX_IND_FACTOR_INC = 0.1
YAW_ANGLE_INC = 5
WIND_SPEED_INC = 2
WIND_DIR_INC = 10

AX_IND_FACTOR_RANGE = (0.1, 0.3)
YAW_ANGLE_RANGE = (-0.20, 0.20)
WIND_SPEED_RANGE = (8, 12)
WIND_DIR_RANGE = (250, 270)

# AX_IND_FACTOR_TEST_POINTS = np.arange(0.1, 0.3 + 0.01, AX_IND_FACTOR_INC)
# YAW_ANGLE_TEST_POINTS = np.arange(-0.15, 0.15 + 0.01, YAW_ANGLE_INC) # np.linspace(--15, 15, N_TEST_POINTS_PER_COORD)
# TURBINE_WIND_SPEED_TEST_POINTS = np.arange(8, 12 + 0.01, TURBINE_WIND_SPEED_INC) #np.linspace(8, 12, N_TEST_POINTS_PER_COORD)
# FREESTREAM_WIND_DIR_TEST_POINTS = np.arange(250, 270 + 0.01, FREESTREAM_WIND_DIR_INC) #np.linspace(250, 270, N_TEST_POINTS_PER_COORD)

class DownstreamTurbineGPR:
    def __init__(self, kernel,
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

        self.X_scaler = None
        self.max_training_size = max_training_size
        self.max_replay_size = max_training_size
        self.gpr = GaussianProcessRegressor(
            kernel=kernel,
            optimizer=optimizer, 
            normalize_y=True, n_restarts_optimizer=10)
        self.turbine_index = turbine_index
        self.upstream_turbine_indices = upstream_turbine_indices
        self.model_type = model_type
        self.X_test = None
    
    def reset_matrices(self):
        self.X_train = np.zeros((0, self.n_inputs))
        self.y_train = np.zeros((0, self.n_outputs))
        self.k_train = []
    
        self.X_train_replay = np.zeros((0, self.n_inputs))
        self.y_train_replay = np.zeros((0, self.n_outputs))
        self.k_train_replay = []
    
    
    def generate_X_test(self, X_current):
        X_test_indices = np.random.randint([N_TEST_POINTS_PER_COORD for l in self.input_labels],
                                           size=(GP_CONSTANTS['N_TEST_POINTS'], len(self.input_labels)))
    
        X_test = []
        for l_idx, l in enumerate(self.input_labels):
            if 'AxIndFactors' in l:
                test_points = np.linspace(max(X_current[l_idx] - AX_IND_FACTOR_INC, AX_IND_FACTOR_RANGE[0]),
                                          min(X_current[l_idx] + AX_IND_FACTOR_INC, AX_IND_FACTOR_RANGE[1]),
                                          N_TEST_POINTS_PER_COORD)
                X_test.append(test_points[X_test_indices[:, l_idx]])
            elif 'YawAngles' in l:
                test_points = np.linspace(max(X_current[l_idx] - YAW_ANGLE_INC, YAW_ANGLE_RANGE[0]),
                                          min(X_current[l_idx] + YAW_ANGLE_INC, YAW_ANGLE_RANGE[1]),
                                          N_TEST_POINTS_PER_COORD)
                X_test.append(test_points[X_test_indices[:, l_idx]])
            elif 'TurbineWindSpeeds' in l:
                test_points = np.linspace(max(X_current[l_idx] - WIND_SPEED_INC, WIND_SPEED_RANGE[0]),
                                          min(X_current[l_idx] + WIND_SPEED_INC, WIND_SPEED_RANGE[1]),
                                          N_TEST_POINTS_PER_COORD)
                X_test.append(test_points[X_test_indices[:, l_idx]])
            elif 'FreestreamWindDir' in l:
                test_points = np.linspace(max(X_current[l_idx] - WIND_DIR_INC, WIND_DIR_RANGE[0]),
                                          min(X_current[l_idx] + WIND_DIR_INC, WIND_DIR_RANGE[1]),
                                          N_TEST_POINTS_PER_COORD)
                X_test.append(test_points[X_test_indices[:, l_idx]])
    
        self.X_test = self.X_scaler.transform(np.transpose(X_test))
    
    def compute_test_var(self):
        _, test_std = self.gpr.predict(self.X_test, return_std=True)
        test_var = np.sum(test_std**2)
        return test_var

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
        effective_dk_to_add = effective_dk.loc[k_idx_to_add]
        history_exists = len(k_to_add) > 0

        return k_to_add, effective_dk, effective_dk_to_add, history_exists
    
    def get_X_current(self, measurements_df, k, effective_dk, k_delay=GP_CONSTANTS['K_DELAY']):
        # TODO
        X_current = self.X_scaler.transform([generate_input_vector(measurements_df, k, self.upstream_turbine_indices, effective_dk.loc[k], k_delay)[1]])
        return X_current
    
    def find_unique_data(self, measurements_df, k_train_potential, effective_dk, k_delay=GP_CONSTANTS['K_DELAY'], y_modeled=None):
        # of all the potential new data points, filter out the unique ones
        X_train_potential = np.vstack([generate_input_vector(
            measurements_df, k, self.upstream_turbine_indices, effective_dk.loc[k], k_delay)[1]
                                       for k in k_train_potential])

        X_current = X_train_potential[np.argmax(k_train_potential), :]
        X_train_potential = self.X_scaler.transform(X_train_potential)
        
        y_train_potential = (
            measurements_df.loc[k_train_potential, f'TurbineWindSpeeds_{self.turbine_index}'].to_numpy() -
            y_modeled).reshape(-1, self.n_outputs)
        
        
        _, unq_idx = np.unique(np.hstack([X_train_potential, y_train_potential]), axis=0,
                               return_index=True)
        X_train_potential = X_train_potential[unq_idx, :]
        y_train_potential = y_train_potential[unq_idx, :]
        k_train_potential = [k_train_potential[i] for i in unq_idx]
        
        # filter out data points that are already contained in replay buffer or training data
        keep_idx = []
        for idx, (X, y, k) in enumerate(zip(X_train_potential,
                                            y_train_potential,
                                            k_train_potential)):
            
            # if this data point matches with any in replay buffer, don't keep
            y_replay_close = np.isclose(self.y_train_replay, y)
            if np.any(y_replay_close) and \
                np.any(np.all(np.isclose(X, self.X_train_replay[np.where(y_replay_close.squeeze())[0], :]), 1), 0):
                continue
            
            # if this data point matches with any in training dataset, don't keep
            y_train_close = np.isclose(self.y_train, y)
            if np.any(y_train_close) and \
                np.any(np.all(np.isclose(X, self.X_train[np.where(y_train_close.squeeze())[0], :]), 1), 0):
                continue
            
            try:
                assert not k in self.k_train # breaking here because Xy_train_potential neq Xy_train where k_train_potential == k_train
            except Exception as e:
                print('oh no')
            # check in choose new data points, that all Xyk are neq
            keep_idx.append(idx)

        X_train_potential = X_train_potential[keep_idx, :]
        y_train_potential = y_train_potential[keep_idx, :]
        k_train_potential = [k_train_potential[i] for i in keep_idx]

        assert len(set(k_train_potential).intersection(self.k_train)) == 0

        # if self.X_scaler is None:
        #     self.X_scaler = StandardScaler().fit(self.X_train_potential)
        #     self.y_scaler = StandardScaler().fit(self.y_train_potential)
            # self.y_train_potential = self.y_scaler.transform(self.y_train_potential)
        
        return X_current, X_train_potential, y_train_potential, k_train_potential
        
    def add_data(self, new_X_train, new_y_train, new_k_train, is_online):
        
        self.X_train = np.vstack([self.X_train, new_X_train])
        self.y_train = np.vstack([self.y_train, new_y_train])
        
        if self.y_train.shape[0] > self.max_training_size:
            self.X_train = self.X_train[-self.max_training_size if self.max_training_size > -1 else 0:,:]
            self.y_train = self.y_train[-self.max_training_size if self.max_training_size > -1 else 0:,:]
        
        if is_online:
            self.k_train = (self.k_train + list(new_k_train))[-self.max_training_size if self.max_training_size > -1
                                                              else 0:]
            assert len(set(self.k_train_replay).intersection(self.k_train)) == 0
        else:
            self.k_train = [-1] * self.X_train.shape[0]

        assert self.X_train.shape[0] <= self.max_training_size
        assert self.k_train.__len__() == self.X_train.shape[0]

    def fit(self):
        return self.gpr.fit(self.X_train, self.y_train)
    
    def save(self):
        with open(os.path.join(DATA_DIR, f'gpr_{self.turbine_index}'), 'wb') as f:
            pickle.dump(self, f)
    
    def predict(self, measurements_df, system_fi, y_modeled=None,
                k_delay=GP_CONSTANTS['K_DELAY'], dt=GP_CONSTANTS['DT']):
        
        time_step_data = measurements_df.iloc[-1]
        k = int(time_step_data['Time'] // dt) # measurements_df.index[-1]
        
        effective_dk = self.compute_effective_dk(system_fi, time_step_data, k_delay=k_delay, dt=dt)
        if k >= (k_delay * effective_dk):
            
            X_ip = generate_input_vector(
                    measurements_df, k, self.upstream_turbine_indices, effective_dk, k_delay)[1]
            X_ip = self.X_scaler.transform(np.atleast_2d(X_ip))
            y_pred, y_std = self.gpr.predict(X_ip, return_std=True)
            y_pred = y_pred.squeeze()
            y_std = y_std.squeeze()
            
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
        # assert self.X_train.shape[0] <= self.max_training_size
        
        # drop BATCH_SIZE datapoints from training dataset at random
        
        init_n_train = int(self.X_train.shape[0])
        keep_idx = list(range(self.X_train.shape[0]))
        np.random.shuffle(keep_idx)
        drop_idx = []
        for _ in range(n_datapoints):
            if len(keep_idx):
                drop_idx.append(keep_idx.pop())
            else:
                break

        assert len(set(k_train_potential).intersection(self.k_train)) == 0
        
        dropped_X_train = self.X_train[drop_idx, :]
        dropped_y_train = self.y_train[drop_idx, :]
        dropped_k_train = [self.k_train[i] for i in drop_idx]
        
        # TODO there are duplicate vals in k_train_potential and dropped_k_train that don't have matching y vals??
        self.X_train_replay = np.vstack([self.X_train_replay, X_train_potential, dropped_X_train])
        self.y_train_replay = np.vstack([self.y_train_replay, y_train_potential, dropped_y_train])
        self.k_train_replay = self.k_train_replay + k_train_potential + dropped_k_train
        assert len(set(k_train_potential).intersection(dropped_k_train)) == 0
        
        # retrain gp with reduced training dataset
        self.X_train = self.X_train[keep_idx, :]
        self.y_train = self.y_train[keep_idx, :]
        self.k_train = [self.k_train[i] for i in keep_idx]

        if len(keep_idx):
            self.fit()
        
        # predict variance for each candidate datapoint in replay buffer
        new_dps = []
        assert len(np.unique(self.k_train_replay)) == len(self.k_train_replay)
        assert len(set(self.k_train_replay).intersection(self.k_train)) == 0
        for i, k in enumerate(self.k_train_replay):
            X = self.X_train_replay[i, :]
            y = self.y_train_replay[i]
            _, std = self.gpr.predict([X], return_std=True)
            new_dps.append((X, y, std[0], k))

        # assign probability of data point being selected to exponential of standard deviation => points with higher standard deviation prediction will be favoured
        new_std_train = [tup[2] for tup in new_dps]
        p_choice = np.exp(new_std_train) / sum(np.exp(new_std_train)) # soft-max exponential
        
        # very large number for probability of choice maps to nan
        nan_idx, = np.where(np.isnan(p_choice))
        if len(nan_idx):
            p_choice = np.zeros_like(p_choice)
            uni_prob = 1.0 / len(nan_idx)
            for i in nan_idx:
                p_choice[i] = uni_prob
        
        # if number of nonzero probabilities < n_datapoints
        assert len(new_dps) == len(p_choice)
        nonzero_p_choice_idx, = np.nonzero(p_choice)
        idx_choice = np.random.choice(list(range(len(new_dps))), min(n_datapoints + len(drop_idx), len(nonzero_p_choice_idx)),
                                      replace=False, p=p_choice)

        if len(idx_choice) < min(n_datapoints + len(drop_idx), len(new_dps)): # multiply by 2 since we dropped n_datapoints from training dataset to add to replay buffer
            # indices of zero probability elements
            zero_p_choice_idx = [i for i in range(len(p_choice)) if i not in nonzero_p_choice_idx]
            
            # select remaining data points from zero probabilities with uniform probability
            add_idx = np.random.choice(zero_p_choice_idx, min(n_datapoints + len(drop_idx), len(new_dps)) - len(idx_choice), replace=False)
            idx_choice = np.append(idx_choice, add_idx)

        # remove from replay buffer data that has been added to training dataset
        idx_left = [idx for idx in range(len(self.k_train_replay)) if idx not in idx_choice]
        self.X_train_replay = self.X_train_replay[idx_left, :]
        self.y_train_replay = self.y_train_replay[idx_left, :]
        self.k_train_replay = [self.k_train_replay[idx] for idx in idx_left]
            
        # add to training data
        new_Xy_train = [new_dps[i] for i in idx_choice]
        new_X_train = np.vstack([tup[0] for tup in new_Xy_train])
        new_y_train = np.vstack([tup[1] for tup in new_Xy_train])
        new_k_train = [tup[3] for tup in new_Xy_train]
        assert len(np.unique(new_k_train)) == len(new_k_train)

        assert new_X_train.shape[0] == min(n_datapoints + len(drop_idx), len(new_dps))

        self.add_data(new_X_train, new_y_train, new_k_train, is_online=True)

        assert (self.X_train.shape[0] == self.max_training_size) \
               or (self.X_train.shape[0] == min(n_datapoints + init_n_train, len(idx_choice) + init_n_train - len(drop_idx)))
        assert self.k_train.__len__() == self.X_train.shape[0]
        assert len(np.unique(self.k_train)) == len(self.k_train)
        assert len(set(self.k_train_replay).intersection(self.k_train)) == 0
        
        # truncate the replay buffer randomly
        shuffle_idx = list(range(len(self.k_train_replay)))
        np.random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[:self.max_replay_size]
        self.X_train_replay = self.X_train_replay[shuffle_idx, :]
        self.y_train_replay = self.y_train_replay[shuffle_idx, :]
        self.k_train_replay = [self.k_train_replay[i] for i in shuffle_idx]
        assert len(set(self.k_train_replay).intersection(self.k_train)) == 0

def optimizer(fun, initial_theta, bounds):
    res = minimize(fun, initial_theta, 
                   method="L-BFGS-B", jac=True, 
                   bounds=bounds, 
                   options={'maxiter': 1000})
    theta_opt = res.x
    func_min = res.fun
    return theta_opt, func_min

# if __name__ == '__main__':
def get_system_info(floris_dir):
    
    ## GET SYSTEM INFORMATION
    system_fi = DynFlorisInterface(floris_dir)
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
 
def get_base_model(floris_dir):

    ## DEFINE PRIOR MODEL
    model_fi = DynFlorisInterface(floris_dir)

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

def get_dfs(df_indices, ts_data_dir, proportion_training_data=GP_CONSTANTS['PROPORTION_TRAINING_DATA']):
    ## FETCH RAW DATA
    # for dir in [DATA_DIR, FIG_DIR]:
    #     if not os.path.exists(dir):
    #         os.mkdir(dir)

    csv_paths = get_paths(ts_data_dir, df_indices=df_indices)

    wake_field_dfs = collect_raw_data(ts_data_dir, csv_paths)
    n_training_datasets = int(np.floor(len(wake_field_dfs) * proportion_training_data))
    full_idx = np.arange(len(wake_field_dfs))
    np.random.shuffle(full_idx)
    training_idx = full_idx[:n_training_datasets]
    testing_idx = full_idx[n_training_datasets:]
    wake_field_dfs = {'train': [wake_field_dfs[i] for i in training_idx],
                      'test': [wake_field_dfs[i] for i in testing_idx]}
    return wake_field_dfs


def init_gprs(system_fi, kernel, k_delay,
              max_training_size=GP_CONSTANTS['MAX_TRAINING_SIZE']):
    ## GENERATE GP MODELS FOR EACH DOWNSTREAM TURBINE'S WIND SPEED
    gprs = []
    for ds_t_idx in system_fi.downstream_turbine_indices:
        # include all turbines with upstream radius of this one
        upstream_turbine_indices = [us_t_idx for us_t_idx in system_fi.upstream_turbine_indices 
                                    if (norm([system_fi.floris.farm.turbine_map.coords[us_t_idx].x1
                                              - system_fi.floris.farm.turbine_map.coords[ds_t_idx].x1,
                                             system_fi.floris.farm.turbine_map.coords[us_t_idx].x2
                                              - system_fi.floris.farm.turbine_map.coords[ds_t_idx].x2
                                             ], ord=2) <= GP_CONSTANTS['UPSTREAM_RADIUS'])
                                    and system_fi.floris.farm.turbine_map.coords[us_t_idx].x1 < system_fi.floris.farm.turbine_map.coords[ds_t_idx].x1]
         
        # PARAMETERIZE GAUSSIAN PROCESS REGRESSOR
         # DotProduct() + WhiteKernel()
        input_labels = generate_input_labels(upstream_turbine_indices, k_delay)
        turbine_input_labels = [l for l in input_labels
                                if 'TurbineWindSpeeds' not in l or int(l.split('_')[1]) in upstream_turbine_indices]

        gpr = DownstreamTurbineGPR(kernel,
                                   max_training_size=max_training_size, 
                                   input_labels=turbine_input_labels, n_outputs=1,
                                   turbine_index=ds_t_idx,
                                   upstream_turbine_indices=upstream_turbine_indices,
                                   model_type=GP_CONSTANTS['MODEL_TYPE'])

        

        gprs.append(gpr)
    return gprs
