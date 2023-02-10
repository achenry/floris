# TODO
# Write .csv data files at checkpoints
# Run DEBUG locally and generate results
# Write Results, Conclusions
# Run DEBUG on RC
# Run not DEBUG on RC and generate results

import numpy as np
from DownstreamTurbineGPR import GP_CONSTANTS, get_system_info, init_gprs, \
    get_base_model, get_dfs, \
    N_TEST_POINTS_PER_COORD, \
    AX_IND_FACTOR_TEST_POINTS, YAW_ANGLE_TEST_POINTS, UPSTREAM_WIND_DIR_TEST_POINTS, UPSTREAM_WIND_SPEED_TEST_POINTS
import pandas as pd
from collections import defaultdict
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from preprocessing import add_gaussian_noise
import multiprocessing as mp
from multiprocessing import Pool
from postprocessing import plot_training_data, plot_score, plot_std_evolution, plot_ts, plot_error_ts, plot_k_train_evolution, compute_score
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
import sys
from floridyn.tools.visualization import plot_turbines_with_fi
import argparse

if sys.platform == 'darwin':
    FARM_LAYOUT = '9turb'
    SIM_SAVE_DIR = f'./{FARM_LAYOUT}_wake_field_simulations'
    TS_SAVE_DIR = f'./{FARM_LAYOUT}_wake_field_tsdata'
    FLORIS_DIR = f'./{FARM_LAYOUT}_floris_input.json'
    BASE_MODEL_FLORIS_DIR = f'./{FARM_LAYOUT}_base_model_floris_input.json'
    DATA_DIR = './data'
    FIG_DIR = './figs'
    SCALARS_DIR = './scalars'
elif sys.platform == 'linux':
    FARM_LAYOUT = '9turb'
    SIM_SAVE_DIR = f'/scratch/alpine/aohe7145/wake_gp/{FARM_LAYOUT}_wake_field_simulations'
    TS_SAVE_DIR = f'/scratch/alpine/aohe7145/wake_gp/{FARM_LAYOUT}_wake_field_ts_data'
    FLORIS_DIR = f'./{FARM_LAYOUT}_floris_input.json'
    BASE_MODEL_FLORIS_DIR = f'./{FARM_LAYOUT}_base_model_floris_input.json'
    DATA_DIR = f'/scratch/alpine/aohe7145/wake_gp/data'
    FIG_DIR = f'/scratch/alpine/aohe7145/wake_gp/figs'
    SCALARS_DIR = '/scratch/alpine/aohe7145/wake_gp//scalars'

FIGSIZE = (30, 21)
COLOR_1 = 'darkgreen'
COLOR_2 = 'indigo'
BIG_FONT_SIZE = 66
SMALL_FONT_SIZE = 62
mpl.rcParams.update({'font.size': SMALL_FONT_SIZE,
					 'axes.titlesize': BIG_FONT_SIZE,
					 'figure.figsize': FIGSIZE,
					 'legend.fontsize': SMALL_FONT_SIZE,
					 'xtick.labelsize': SMALL_FONT_SIZE,
					 'ytick.labelsize': SMALL_FONT_SIZE,
                     'lines.linewidth': 4,
					 'figure.autolayout': True,
                     'lines.markersize':10})

TRAIN_ONLINE = True
TRAIN_OFFLINE = False
FIT_ONLINE = True
GP_CONSTANTS['PROPORTION_TRAINING_DATA'] = 1

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', default=False)
parser.add_argument('-p', '--parallel', action='store_true', default=False)
parser.add_argument('-gc', '--generate_scalars', action='store_true', default=False)
parser.add_argument('-rs', '--run_simulations', action='store_true', default=False)
parser.add_argument('-gp', '--generate_plots', action='store_true', default=False)
parser.add_argument('case_ids', type=int, nargs='+')
args = parser.parse_args()
DEBUG = args.debug
PARALLEL = args.parallel
GENERATE_SCALARS = args.generate_scalars
RUN_SIMULATIONS = args.run_simulations
GENERATE_PLOTS  = args.generate_plots

# TODO may need more variation in training datasets

TMAX = 1200 #300 if DEBUG else 1200
N_TOTAL_DATASETS = 4 if DEBUG else 500

# construct case hierarchy
KERNELS = [lambda: ConstantKernel(constant_value_bounds=(1e-12, 1e12)) * RBF(length_scale_bounds=(1e-12, 1e12)),
           lambda: ConstantKernel(constant_value_bounds=(1e-12, 1e12)) * Matern(length_scale_bounds=(1e-12, 1e12))]
MAX_TRAINING_SIZE_VALS = [20, 40, 80, 160]
NOISE_STD_VALS = [0.0001, 0.001, 0.01, 0.1]
K_DELAY_VALS = [2, 4, 6, 8]
BATCH_SIZE_VALS = [1, 2, 3, 4]
default_kernel = lambda: ConstantKernel(constant_value_bounds=(1e-12, 1e12)) * RBF(length_scale_bounds=(1e-12, 1e12))
default_kernel_idx = 0
default_max_training_size = 40
default_batch_size = 2
default_noise_std = 0.01
default_k_delay = 4

cases = [{'kernel': default_kernel(), 'max_training_size': default_max_training_size,
          'noise_std': default_noise_std, 'k_delay': default_k_delay, 'batch_size': x} for x in BATCH_SIZE_VALS] + \
        [{'kernel': default_kernel(), 'max_training_size': default_max_training_size, 'noise_std': default_noise_std,
          'k_delay': x, 'batch_size': default_batch_size} for x in K_DELAY_VALS if x != default_k_delay] + \
        [{'kernel': default_kernel(), 'max_training_size': default_max_training_size, 'noise_std': x,
          'k_delay': default_k_delay, 'batch_size': default_batch_size} for x in NOISE_STD_VALS if x != default_noise_std] + \
        [{'kernel': default_kernel(), 'max_training_size': x, 'noise_std': default_noise_std,
          'k_delay': default_k_delay, 'batch_size': default_batch_size} for x in MAX_TRAINING_SIZE_VALS if x != default_max_training_size] + \
        [{'kernel': x(), 'max_training_size': default_max_training_size, 'noise_std': default_noise_std,
          'k_delay': default_k_delay, 'batch_size': default_batch_size} for i, x in enumerate(KERNELS) if i != default_kernel_idx]

for case_idx, case in enumerate(cases):
    print(f'\nCase {case_idx + 1}')
    print(f'N_tr = {case["max_training_size"]}')
    print(f'kernel = {case["kernel"]}')
    print(f'noise_std = {case["noise_std"]}')
    print(f'k_delay = {case["k_delay"]}')
    print(f'batch_size = {case["batch_size"]}')

CASE_IDS = args.case_ids if -1 not in args.case_ids else list(range(len(cases)))
cases = [cases[c] if c in CASE_IDS else None for c in range(len(cases))]

if not os.path.exists(os.path.join(SIM_SAVE_DIR)):
    os.makedirs(SIM_SAVE_DIR)

if not os.path.exists(os.path.join(FIG_DIR)):
    os.makedirs(FIG_DIR)
    
if not os.path.exists(os.path.join(SCALARS_DIR)):
    os.makedirs(SCALARS_DIR)

def initialize(case_idx, full_offline_measurements_df, system_fi, k_delay, noise_std, max_training_size, kernel,
               n_test_points=GP_CONSTANTS['N_TEST_POINTS'], train_offline=False, scalar_dir=None):
    """
    
    Args:
        case_idx:
        full_offline_measurements_df:
        system_fi:
        k_delay:
        noise_std:
        max_training_size:
        kernel:
        n_test_points:

    Returns:

    """
    print(f'Initializing GPs for Case {case_idx}')
    gprs = init_gprs(system_fi, kernel=kernel, k_delay=k_delay, max_training_size=max_training_size)

    # add noise to Turbine Wind Speed measurements
    if scalar_dir is None or not os.path.exists(scalar_dir) or train_offline:
        noisy_measurements_df = add_gaussian_noise(system_fi, full_offline_measurements_df, std=noise_std)
        # simulate modeled effective wind speeds at downstream turbines for measurements

        if GP_CONSTANTS['MODEL_TYPE'] == 'error':
            # initialize to steady-state
            model_fi.floris.farm.flow_field.mean_wind_speed = noisy_measurements_df.loc[0, 'FreestreamWindSpeed']
            model_fi.reinitialize_flow_field(
                wind_speed=noisy_measurements_df.loc[0, 'FreestreamWindSpeed'],
                wind_direction=noisy_measurements_df.loc[0, 'FreestreamWindDir'])
            model_fi.calculate_wake(
                yaw_angles=[noisy_measurements_df.loc[0, f'YawAngles_{t_idx}']
                            for t_idx in model_fi.turbine_indices],
                axial_induction=[noisy_measurements_df.loc[0, f'AxIndFactors_{t_idx}']
                                 for t_idx in model_fi.turbine_indices])
        
            y_modeled = []
            for k in noisy_measurements_df.index:
                sim_time = noisy_measurements_df.loc[k, 'Time']
                model_fi.floris.farm.flow_field.mean_wind_speed = noisy_measurements_df.loc[k, 'FreestreamWindSpeed']
                model_fi.reinitialize_flow_field(
                    wind_speed=noisy_measurements_df.loc[k, 'FreestreamWindSpeed'],
                    wind_direction=noisy_measurements_df.loc[k, 'FreestreamWindDir'],
                    sim_time=sim_time)
                model_fi.calculate_wake(
                    yaw_angles=[noisy_measurements_df.loc[k, f'YawAngles_{t_idx}']
                                for t_idx in model_fi.turbine_indices],
                    axial_induction=[noisy_measurements_df.loc[k, f'AxIndFactors_{t_idx}']
                                     for t_idx in model_fi.turbine_indices],
                    sim_time=sim_time)
                y_modeled = y_modeled + [[model_fi.floris.farm.turbines[gp.turbine_index].average_velocity for gp in gprs]]
                
            y_modeled = np.vstack(y_modeled)
            # noisy_measurements_df[f'TurbineWindSpeeds_{gp.turbine_index}'].to_numpy()
        else:
            y_modeled = np.zeros(len(noisy_measurements_df.index), len(gprs))
        
    # for each downstream wind turbine
    for gp_idx, gp in enumerate(gprs):
        
        if scalar_dir is None or not os.path.exists(scalar_dir) or train_offline:
            k_to_add, effective_dk, reduced_effective_dk, history_exists = \
                gp.check_history(noisy_measurements_df, system_fi, k_delay=k_delay, dt=GP_CONSTANTS['DT'])

            # print(noisy_measurements_df.index[-1], effective_dk, gp.turbine_index)

            if not history_exists:
                continue

            # shuffle order to give random data points to GP, then truncate to number of data points needed
            shuffle_idx = list(k_to_add.index)
            np.random.shuffle(shuffle_idx)
            shuffle_idx = shuffle_idx[:max_training_size]

            # compute the base model values for the wake, given all of the freestream wind speeds, yaw angles and axial induction factors over time for each time-series dataset
            X_train_new, y_train_new, _ = gp.prepare_data(noisy_measurements_df, k_to_add.loc[shuffle_idx],
                                                       reduced_effective_dk.loc[shuffle_idx],
                                                       y_modeled=y_modeled[shuffle_idx, gp_idx],
                                                       k_delay=k_delay)

            if train_offline:
                gp.add_data(X_train_new, y_train_new, k_to_add.loc[shuffle_idx], is_online=False)

                # Fit data
                gp.fit()
                
        elif scalar_dir is not None and os.path.exists(scalar_dir):
            with open(os.path.join(scalar_dir, f'X_scalar_case{case_idx}_gp{gp_idx}'), 'rb') as fp:
                gp.X_scalar = pickle.load(fp)
            with open(os.path.join(scalar_dir, f'y_scalar_case{case_idx}_gp{gp_idx}'), 'rb') as fp:
                gp.y_scalar = pickle.load(fp)
        
        X_test_indices = np.random.randint([N_TEST_POINTS_PER_COORD for l in gp.input_labels],
                                           size=(n_test_points, len(gp.input_labels)))

        X_test = []
        for l_idx, l in enumerate(gp.input_labels):
            if 'AxIndFactors' in l:
                X_test.append(AX_IND_FACTOR_TEST_POINTS[X_test_indices[:, l_idx]])
            elif 'YawAngles' in l:
                X_test.append(YAW_ANGLE_TEST_POINTS[X_test_indices[:, l_idx]])
            elif 'TurbineWindSpeeds' in l:
                X_test.append(UPSTREAM_WIND_SPEED_TEST_POINTS[X_test_indices[:, l_idx]])
            elif 'FreestreamWindDir' in l:
                X_test.append(UPSTREAM_WIND_DIR_TEST_POINTS[X_test_indices[:, l_idx]])

        gp.X_test = gp.X_scalar.transform(np.transpose(X_test))

    return gprs

def run_single_simulation(case_idx, gprs, simulation_df, simulation_idx,
                          current_input_labels, k_delay, noise_std, batch_size, dataset_type):
    
    print(f'Running Simulation {simulation_idx} for Case {case_idx}')
    
    # Fetch wind farm system layout information, floris interface used to simulate 'true' wind farm
    system_fi = get_system_info(FLORIS_DIR)

    # Reset base wind farm model i.e. assumed wake model, where GP learns error between measurements and model
    model_fi = get_base_model(BASE_MODEL_FLORIS_DIR)

    print('Defining wind, yaw, ax ind factor disturbances')

    disturbances = {
        'FreestreamWindSpeed': simulation_df['FreestreamWindSpeed'],
        'FreestreamWindDir': simulation_df['FreestreamWindDir']
    }

    for t_idx in system_fi.turbine_indices:
        disturbances[f'YawAngles_{t_idx}'] = simulation_df[f'YawAngles_{t_idx}']
        disturbances[f'AxIndFactors_{t_idx}'] = simulation_df[f'AxIndFactors_{t_idx}']

    online_measurements_df = pd.DataFrame(columns=current_input_labels) # holds all current and historic time-step data needed to make autoregressive inputs
    # k_buffered = [] # list of discrete time-steps for which the data point has not been added to the training data yet for lack of history for the autoregressive inputs
    # replay_kXy_train = {'k': [], 'X': [], 'y': []}

    kmax_gp = int(TMAX // GP_DT)
    y_true = [[] for i in range(kmax_gp)]
    y_modeled = [[] for i in range(kmax_gp)]
    y_pred = [[] for i in range(kmax_gp)]
    y_std = [[] for i in range(kmax_gp)]
    y_meas = [[] for i in range(kmax_gp)]
    y_train_frames = []
    k_train_frames = [[] for i in range(kmax_gp)]
    test_std = np.nan * np.ones((kmax_gp, len(gprs)))
    # test_rmse = np.nan * np.ones((kmax_gp, len(gprs)))

    # initialize system simulator and model simulator to steady-state
    for fi in [system_fi, model_fi]:
        fi.floris.farm.flow_field.mean_wind_speed = disturbances['FreestreamWindSpeed'][0]
        fi.reinitialize_flow_field(wind_speed=disturbances['FreestreamWindSpeed'][0], wind_direction=disturbances['FreestreamWindDir'][0])
        fi.calculate_wake(yaw_angles=[disturbances[f'YawAngles_{t_idx}'][0] for t_idx in system_fi.turbine_indices],
                axial_induction=[disturbances[f'AxIndFactors_{t_idx}'][0] for t_idx in system_fi.turbine_indices])

    for k_sys, t in enumerate(range(0, TMAX, SYS_DT)):
        print(f'Simulating {k_sys}th Time-Step')
        
        # step simulation of wake farm forward ('true' values) - system_fi
        # step simulation of base wake model forward ('model' values for computing error) - model_fi
        for fi in [system_fi, model_fi]:
            fi.floris.farm.flow_field.mean_wind_speed = disturbances['FreestreamWindSpeed'][k_sys]

            fi.reinitialize_flow_field(
                wind_speed=disturbances['FreestreamWindSpeed'][k_sys],
                wind_direction=disturbances['FreestreamWindDir'][k_sys],
                sim_time=t)

            fi.calculate_wake(
                yaw_angles=[disturbances[f'YawAngles_{t_idx}'][k_sys] for t_idx in fi.turbine_indices],
                axial_induction=[disturbances[f'AxIndFactors_{t_idx}'][k_sys] for t_idx in fi.turbine_indices],
                sim_time=t)

        # if GP sampling time has passed
        if t % GP_DT == 0:
            
            # time-step of learning procedure
            k_gp = int(t // GP_DT)

            # for each downstream turbine
            for gp_idx, ds_idx in enumerate(system_fi.downstream_turbine_indices):
                
                # modeled effective wind speed at this ds turbine from base_model
                y_modeled_k = model_fi.floris.farm.turbines[ds_idx].average_velocity \
                    if GP_CONSTANTS['MODEL_TYPE'] == 'error' else 0
                y_modeled[k_gp].append(y_modeled_k)
                
                # true effective wind speed at this ds turgbine
                y_true_k = system_fi.floris.farm.turbines[ds_idx].average_velocity
                y_true[k_gp].append(y_true_k)

            print(f'Adding {k_gp}th GP Sample')

            # collect current measurements (at this time-step k_gp) in buffer: time
            current_measurements = defaultdict(list)
            current_measurements['Time'].append(t)

            # collect measurements: freestream wind speed and direction
            current_measurements['FreestreamWindSpeed'].append(disturbances['FreestreamWindSpeed'][k_sys])
            current_measurements['FreestreamWindDir'].append(disturbances['FreestreamWindDir'][k_sys])

            # for each upstream turbine
            for us_turbine_idx in system_fi.upstream_turbine_indices:
                # collect new measurements: axial induction factor, yaw angle (inputs)
                current_measurements[f'TurbineWindSpeeds_{us_turbine_idx}'].append(
                    system_fi.floris.farm.turbines[us_turbine_idx].average_velocity)
                current_measurements[f'YawAngles_{us_turbine_idx}'].append(
                    system_fi.floris.farm.turbines[us_turbine_idx].yaw_angle)
                current_measurements[f'AxIndFactors_{us_turbine_idx}'].append(
                    system_fi.floris.farm.turbines[us_turbine_idx].ai_set)

            # for each downstream turbine
            for gp_idx, ds_turbine_idx in enumerate(system_fi.downstream_turbine_indices):
                if ds_turbine_idx not in system_fi.upstream_turbine_indices:
                    # collect new measurements: effective wind speed at downstream turbine (target)
                    current_measurements[f'TurbineWindSpeeds_{ds_turbine_idx}'].append(
                        system_fi.floris.farm.turbines[ds_turbine_idx].average_velocity)

            # add Gaussian noise to effective wind speed measurements
            current_measurements = pd.DataFrame(add_gaussian_noise(system_fi, current_measurements, std=noise_std))

            # add collected data to online training buffer,
            online_measurements_df = pd.concat([online_measurements_df, current_measurements],
                                               ignore_index=True)

            # list of 'current' time-steps buffered in online_measurements_df
            # k_buffered.append(k_gp)
            # assert online_measurements_df['Time'].iloc[-1] == k_gp
            # assert (np.diff(online_measurements_df.index) == 1).all()

            # if enough samples have been added to online_measurements_df to make a batch
            if len(online_measurements_df.index) >= batch_size and TRAIN_ONLINE:

                # drop the historic inputs we no longer need for ANY downstream turbine
                # i.e. keep the last max(effective_dk * k_delay) for all downstream turbines
                # IF we used some of the current measurements
                min_k_needed = set()
                
                y_train_frames.append([])
                

                # for each downstream turbine
                for gp_idx, ds_turbine_idx in enumerate(system_fi.downstream_turbine_indices):

                    # plot evolution of gprs[gp_idx].y_train
                    if len(gprs[gp_idx].y_train):
                        y_train_unique, y_train_count = np.unique(gprs[gp_idx].y_train, return_counts=True)
                        y_train_unique = gprs[gp_idx].y_scalar.inverse_transform(
                            y_train_unique.reshape((-1, 1))).reshape((1, -1))[0]
                        y_train_frames[-1].append(sorted(list(zip(y_train_unique, y_train_count)), key=lambda tup: tup[0]))
                    
                    # k_to_add is a list of the time-steps for which enough historic inputs exist to add to training data
                    k_to_add, effective_dk, effective_dk_to_add, history_exists \
                        = gprs[gp_idx].check_history(online_measurements_df, system_fi,
                                                     k_delay=k_delay,
                                                     dt=GP_CONSTANTS['DT'])

                    # drop time-steps in online_measurements_df for which < effective_dk * k_delay
                    # compute the first k step required for the autoregressive inputs
                    min_k_needed.update(list(
                        online_measurements_df['Time-Step'] - (effective_dk * GP_CONSTANTS['K_DELAY'])))

                    # add new online measurements to existing set of measurements if there exists enough historic measurements
                    # unadded_k is a list of time-steps for which enough historic inputs exist to add to training data AND has not already been added to GPs training data or replay buffer
                    unadded_k_idx = [k_idx for k_idx, k in k_to_add.iteritems() if k not in gprs[gp_idx].k_train_replay + gprs[gp_idx].k_train]
                    unadded_k = k_to_add.loc[unadded_k_idx]
                    
                    # if not history_exists, want to have at least the required batch of new datapoints to choose from
                    
                    if len(unadded_k_idx) < batch_size:
                        # print(f'Not enough history available to fit {gp_idx} th GP with batch of samples, adding to buffer instead')
                        continue

                    # print(f'Enough history available to fit {gp_idx} th GP with {len(unadded_k_idx)} new samples '
                    #       f'between k = {unadded_k.values[0]} and {unadded_k.values[-1]}')
                    potential_X_train_new, potential_y_train_new, potential_k_train_new \
                        = gprs[gp_idx].prepare_data(online_measurements_df,
                                                    unadded_k,
                                                    effective_dk.loc[unadded_k_idx],
                                                    y_modeled=[y_modeled[k][gp_idx] for k in unadded_k.values],
                                                    k_delay=k_delay)
                    
                    if len(potential_k_train_new) < batch_size:
                        print(
                            f'Not enough history available to fit {gp_idx} th GP with batch of samples, adding to buffer instead')
                        continue
                        
                    print(f'Enough history available to fit {gp_idx} th GP with {len(potential_k_train_new)} new samples '
                          f'between k = {potential_k_train_new.values[0]} and {potential_k_train_new.values[-1]}')

                    # assert potential_X_train_new.shape[0] == len(unadded_k)
                    gprs[gp_idx].choose_new_data(potential_X_train_new, potential_y_train_new, potential_k_train_new,
                                                 n_datapoints=batch_size)
                    
                    # TODO why are there duplicates for a given gp at a given time-step
                    # add this GPs data points identifiers for this time-step
                    k_train_frames[k_gp].append(gprs[gp_idx].k_train)
                    
                    # refit gp
                    if FIT_ONLINE:
                        gprs[gp_idx].fit()
                        # TODO why is this not changing?
                        test_std[k_gp, gp_idx] = gprs[gp_idx].compute_test_std()
                        # test_score[k_gp, gp_idx] = gprs[gp_idx].compute_test_rmse()

                min_k_needed = max(min(min_k_needed), 0)
                if min_k_needed > 0:
                    print(f'Removing samples up to k={min_k_needed} from online measurements buffer because no GP needs '
                          f'them for autoregressive inputs.')
                    online_measurements_df = online_measurements_df.loc[min_k_needed:]

            # for each downstream turbine
            for gp_idx, ds_idx in enumerate(system_fi.downstream_turbine_indices):
                # predict effective wind speed
                y_pred_k, y_std_k = gprs[gp_idx].predict(online_measurements_df, system_fi=system_fi,
                                                         k_delay=k_delay, y_modeled=y_modeled_k)
                
                    
                y_pred[k_gp].append(y_pred_k)
                y_std[k_gp].append(y_std_k)

                y_meas_k = current_measurements[f'TurbineWindSpeeds_{ds_idx}']
                y_meas[k_gp].append(y_meas_k[0])

    y_true = np.vstack(y_true)
    y_modeled = np.vstack(y_modeled)
    y_pred = np.vstack(y_pred)
    y_std = np.vstack(y_std)
    y_meas = np.vstack(y_meas)
    test_std = np.vstack(test_std)
    # test_rmse = np.vstack(test_rmse)

    results = {'true': y_true, 'modeled': y_modeled, 'pred': y_pred, 'std': y_std, 'meas': y_meas,
               'test_std': test_std, 'k_train': k_train_frames}
    
    filename = f'simulation_data_{dataset_type}_case-{case_idx}_df-{simulation_idx}'
    with open(os.path.join(SIM_SAVE_DIR, filename), 'wb') as fp:
        pickle.dump(results, fp)

    training_fig, training_ax = plt.subplots(facecolor = plt.cm.Greys(0.2),
                      dpi = 150,
                      tight_layout=True)
    
    # training_max_count = max(max(tup[1] for tup in arr[gp_idx]) if len(arr[gp_idx]) else -1 for gp_idx in range(len(gprs)) for arr in y_train_frames if len(arr))
    training_y_vals = [[tup[0] for tup in arr[gp_idx]] for gp_idx in range(len(gprs)) for arr in y_train_frames if len(arr)]
    training_y_counts = [[tup[1] for tup in arr[gp_idx]] for gp_idx in range(len(gprs)) for arr in y_train_frames if len(arr)]
    # if len(training_y_vals)
    training_y_vals = sorted(set(np.concatenate(training_y_vals))) if len(training_y_vals) else []
    training_max_count = np.max(training_y_counts) if len(training_y_counts) else -1

    def animate(k_idx, *fargs):
        # training_ax = training_fig.add_subplot(1, 1, 1)
        training_ax.clear()
        if training_max_count > -1:
            training_ax.set_ylim([0, training_max_count])
        training_ax.set_xticks(training_y_vals)
        training_ax.set_facecolor(plt.cm.Greys(0.2))
        # [spine.set_visible(False) for spine in training_ax.spines.values()] # remove chart outlines

        gp_idx = fargs[0]
        count_vals = [tup[1] for tup in y_train_frames[k_idx][gp_idx]]
        y_vals = [tup[0] for tup in y_train_frames[k_idx][gp_idx]]
        training_ax.bar(y_vals, count_vals, width=0.05, align='center')

    # for gp_idx, ds_idx in enumerate(system_fi.downstream_turbine_indices):

        # ax.set(title=f'TurbineWindSpeeds_{ds_idx} GP Training Outputs')
    # gp_idx = 0
    # ani = FuncAnimation(training_fig, animate, frames=len(y_train_frames), interval=25, fargs=[gp_idx])
    # ani.save(os.path.join(FIG_DIR, f'training_ani_case-{case_idx}_df-{simulation_idx}_gp.mp4'))

        
    return results


if __name__ == '__main__':
    ## TURBINE WIND SPEED GP

    # Fetch wind farm system layout information, floris interface used to simulate 'true' wind farm
    system_fi = get_system_info(FLORIS_DIR)
    assert system_fi.get_model_parameters()["Wake Deflection Parameters"]["use_secondary_steering"] == False
    assert "use_yaw_added_recovery" not in system_fi.get_model_parameters()["Wake Deflection Parameters"] or \
           system_fi.get_model_parameters()["Wake Deflection Parameters"]["use_yaw_added_recovery"] == False
    assert "calculate_VW_velocities" not in system_fi.get_model_parameters()["Wake Deflection Parameters"] or \
           system_fi.get_model_parameters()["Wake Deflection Parameters"]["calculate_VW_velocities"] == False

    # Fetch base wind farm model i.e. assumed wake model, where GP learns error between measurements and model
    model_fi = get_base_model(BASE_MODEL_FLORIS_DIR)
    assert model_fi.get_model_parameters()["Wake Deflection Parameters"]["use_secondary_steering"] == False
    assert "use_yaw_added_recovery" not in model_fi.get_model_parameters()["Wake Deflection Parameters"] or \
           model_fi.get_model_parameters()["Wake Deflection Parameters"]["use_yaw_added_recovery"] == False
    assert "calculate_VW_velocities" not in model_fi.get_model_parameters()["Wake Deflection Parameters"] or \
           model_fi.get_model_parameters()["Wake Deflection Parameters"]["calculate_VW_velocities"] == False
    
    # Read raw offline training data
    print('Reading time-series datasets')

    # consider the following time-series cases
    df_indices = N_TOTAL_DATASETS # np.random.randint(0, 80, 5) #list(range(20))s
    wake_field_dfs = get_dfs(df_indices, TS_SAVE_DIR, proportion_training_data=GP_CONSTANTS['PROPORTION_TRAINING_DATA'])
    
    full_offline_measurements_df = pd.concat(wake_field_dfs['train'], ignore_index=True) \
        if TRAIN_OFFLINE or GENERATE_SCALARS else None

    # time, X, y, current_input_labels, input_labels \
    if PARALLEL and RUN_SIMULATIONS:
        pool = Pool(mp.cpu_count())
        case_gprs_tmp = pool.starmap(initialize,
                                     [(case_idx, full_offline_measurements_df, system_fi,
                                       case['k_delay'], case['noise_std'], case['max_training_size'],
                                       case['kernel'],
                                       GP_CONSTANTS['N_TEST_POINTS'], TRAIN_OFFLINE,
                                       SCALARS_DIR if not GENERATE_SCALARS else None)
                                      for case_idx, case in enumerate(cases) if case is not None])
        case_gprs = []
        for case in cases:
            if case is not None:
                case_gprs.append(case_gprs_tmp.pop(0))
            else:
                case_gprs.append(None)
        pool.close()
    elif RUN_SIMULATIONS:
        case_gprs = []
        for case_idx, case in enumerate(cases):
            if case is not None:
                case_gprs.append(initialize(case_idx, full_offline_measurements_df, system_fi,
                           case['k_delay'], case['noise_std'], case['max_training_size'], case['kernel'],
                                            GP_CONSTANTS['N_TEST_POINTS'], TRAIN_OFFLINE, SCALARS_DIR if not GENERATE_SCALARS else None))
            else:
                case_gprs.append(None)
        
    if GENERATE_SCALARS:
        for case_idx, case in enumerate(case_gprs):
            if case is not None:
                for gp_idx, gp in enumerate(case):
                    with open(os.path.join(SCALARS_DIR, f'X_scalar_case{case_idx}_gp{gp_idx}'), 'wb') as fp:
                        pickle.dump(gp.X_scalar, fp)
                    with open(os.path.join(SCALARS_DIR, f'y_scalar_case{case_idx}_gp{gp_idx}'), 'wb') as fp:
                        pickle.dump(gp.y_scalar, fp)

    # start simulation
    GP_DT = GP_CONSTANTS['DT']
    SYS_DT = 1
    K_SYS_MAX = int(TMAX // SYS_DT)
    time_steps = range(K_SYS_MAX)

    current_input_labels = ['Time', 'Time-Step'] + ['FreestreamWindSpeed', 'FreestreamWindDir'] \
                           + [f'TurbineWindSpeeds_{t}' for t in system_fi.turbine_indices] \
                           + [f'AxIndFactors_{t}' for t in system_fi.turbine_indices] \
                           + [f'YawAngles_{t}' for t in system_fi.turbine_indices]
    
    if PARALLEL and RUN_SIMULATIONS:
        simulation_train_cases = np.concatenate([[(case_idx, case_gprs[case_idx], df, df_idx,
                                                   current_input_labels,
                                                   case['k_delay'], case['noise_std'], case['batch_size'], 'train')
                                                  for df_idx, df in enumerate(wake_field_dfs['train'])]
                                                 for case_idx, case in enumerate(cases) if case is not None])

        pool = Pool(mp.cpu_count())
        simulations_train = pool.starmap(run_single_simulation, simulation_train_cases)
        pool.close()
    elif RUN_SIMULATIONS:
        simulations_train = []
        for case_idx, case in enumerate(cases):
            if case is not None:
                for df_idx, df in enumerate(wake_field_dfs['train']):
                    simulations_train.append(run_single_simulation(case_idx, case_gprs[case_idx], df, df_idx,
                                                   current_input_labels,
                                                   case['k_delay'], case['noise_std'], case['batch_size'], 'train'))
                

    if GENERATE_PLOTS:
        system_fi = get_system_info(FLORIS_DIR)
        ## FETCH SIMULATION RESULTS
        simulation_results = defaultdict(list)
        for root, dir, filenames in os.walk(SIM_SAVE_DIR):
            for filename in sorted(filenames):
                if 'simulation_data' in filename:
                    if 'train' in filename:
                        with open(os.path.join(SIM_SAVE_DIR, filename), 'rb') as fp:
                            simulation_results['train'].append(pickle.load(fp))
                    elif 'test' in filename:
                        with open(os.path.join(SIM_SAVE_DIR, filename), 'rb') as fp:
                            simulation_results['test'].append(pickle.load(fp))

        ## PLOT RESULTS
        # plot the true vs predicted gp values over the course of the simulation

        
        score_type = 'rmse'
        sim_score, turbine_sim_score, turbine_score_mean, turbine_score_std \
            = compute_score(system_fi, simulation_results, score_type=score_type)

        score_fig = plot_score(system_fi, turbine_sim_score, turbine_score_mean, turbine_score_std, score_type)
        score_fig.show()
        score_fig.savefig(os.path.join(FIG_DIR, f'score.png'))

        if len(system_fi.floris.farm.turbines) == 9:
            ds_indices = [4, 7]
        else:
            # for 2 turbine farm
            ds_indices = [1]

        # choose training and test datasets with lowest mean rmse over all turbines
        n_ts_plots = 2
        sim_indices = {'train': np.argsort(sim_score['train'])[:n_ts_plots]}

        ts_fig = plot_ts(system_fi.downstream_turbine_indices, ds_indices, simulation_results, sim_indices, np.arange(0, TMAX, GP_DT))
        ts_fig.show()
        ts_fig.savefig(os.path.join(FIG_DIR, f'time_series.png'))
        
        std_fig = plot_std_evolution(system_fi.downstream_turbine_indices, ds_indices, simulation_results, sim_indices,
                                     np.arange(0, TMAX, GP_DT))
        std_fig.show()
        plt.savefig(os.path.join(FIG_DIR, f'std_evolution.png'))

        k_train_fig = plot_k_train_evolution(system_fi.downstream_turbine_indices, ds_indices, simulation_results,
                                             sim_indices, np.arange(0, TMAX, GP_DT))
        k_train_fig.show()
        plt.savefig(os.path.join(FIG_DIR, f'k_train_evolution.png'))

        error_ts_fig = plot_error_ts(system_fi, simulation_results, sim_indices)
        error_ts_fig.show()
        plt.savefig(os.path.join(FIG_DIR, f'error_ts.png'))
        
        farm_fig, farm_ax = plt.subplots(1, 1)
        plot_turbines_with_fi(farm_ax, system_fi)
        farm_fig.savefig(os.path.join(FIG_DIR, f'wind_farm.png'))

        # def plot_score_evolution():
        #     """
        #     plot evolution of rmse of predicted mean at grid test points for middle column downstream turbines vs online training time
        #     Returns:
        #
        #     """
        #     time = np.arange(TMAX)
        #     rmse_fig, rmse_ax = plt.subplots(len(sim_indices['train']) + len(sim_indices['test']), 1,
        #                                    sharex=True, sharey=True)
        #
        #     ax_idx = -1
        #     for i, dataset_type in enumerate(['train', 'test']):
        #         for j, sim_idx in enumerate(sim_indices[dataset_type]):
        #             ax_idx += 1
        #             for ds_idx in ds_indices:
        #                 rmse_ax[ax_idx].plot(time, simulation_results[sim_idx]['test_rmse'][:, ds_idx],
        #                                     label=f'Turbine {ds_idx}')
        #
        #                 rmse_ax[ax_idx].set(
        #                     title=f'Downstream Turbine Effective Wind Speed RMSE '
        #                           f'vs. Time for {dataset_type.capitalize()}ing Simulation {j} [m/s]')
        #
        #     rmse_ax[0].legend(loc='center left')
        #     rmse_ax[-1].set(xlabel='Time [s]')
        #     rmse_fig.show()
        #     plt.savefig(os.path.join(FIG_DIR, f'rmse_evolution.png'))