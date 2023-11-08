# TODO
# Run DEBUG locally for one wake_field and one case and generate results
# Update Methodology
#   MinMaxScalar for Xinput
# Write Case Study,
#  Figure Captions, Case Descriptions
# Conclusions
# Run DEBUG on RC
# Run not DEBUG on RC and generate results

import numpy as np
import pandas as pd
from collections import defaultdict
import pickle
import os
from preprocessing import add_gaussian_noise, get_paths
import multiprocessing as mp
from multiprocessing import Pool
from postprocessing import plot_score, plot_std_evolution, plot_ts, plot_error_ts, plot_k_train_evolution, \
    compute_scores, compute_errors, plot_wind_farm, generate_scores_table, generate_errors_table
import argparse
from sklearn.preprocessing import MinMaxScaler
import time

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', default=False)
parser.add_argument('-p', '--parallel', action='store_true', default=False)
parser.add_argument('-gc', '--generate_scalers', action='store_true', default=False)
parser.add_argument('-rs', '--run_simulations', action='store_true', default=False)
parser.add_argument('-gp', '--generate_plots', action='store_true', default=False)
parser.add_argument('case_ids', type=int, nargs='+')
args = parser.parse_args()
DEBUG = args.debug
PARALLEL = args.parallel
GENERATE_SCALARS = args.generate_scalers
RUN_SIMULATIONS = args.run_simulations
GENERATE_PLOTS  = args.generate_plots

TMAX = 3600 #300 if DEBUG else 1200
N_TOTAL_DATASETS = 9 if DEBUG else 500

# construct case hierarchy

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

def initialize(case_idx, system_fi, k_delay, max_training_size, kernel):
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

    # for each downstream wind turbine, set MinMaxScaler up
    for gp_idx, gp in enumerate(gprs):
        gp.X_scaler = MinMaxScaler()
        X_range = [[], []]

        for l_idx, l in enumerate(gp.input_labels):
            if 'AxIndFactors' in l:
                X_range[0].append(AX_IND_FACTOR_RANGE[0])
                X_range[1].append(AX_IND_FACTOR_RANGE[1])
            elif 'YawAngles' in l:
                X_range[0].append(YAW_ANGLE_RANGE[0])
                X_range[1].append(YAW_ANGLE_RANGE[1])
            elif 'TurbineWindSpeeds' in l:
                X_range[0].append(WIND_SPEED_RANGE[0])
                X_range[1].append(WIND_SPEED_RANGE[1])
            elif 'FreestreamWindDir' in l:
                X_range[0].append(WIND_DIR_RANGE[0])
                X_range[1].append(WIND_DIR_RANGE[1])
         
        gp.X_scaler.fit(X_range)
        
    return gprs

def run_single_simulation(case_idx, gprs, simulation_df_path, simulation_idx,
                          current_input_labels, k_delay, noise_std, batch_size, max_training_size, dataset_type):
     
    print(f'Running Simulation {simulation_idx} for Case {case_idx}')
    simulation_df = pd.read_csv(simulation_df_path, index_col=0)
    
    mean_wind_speed = simulation_df['FreestreamWindSpeed'].mean()
    mean_wind_dir = simulation_df['FreestreamWindDir'].mean()
    
    # Fetch wind farm system layout information, floris interface used to simulate 'true' wind farm
    system_fi = get_system_info(FLORIS_DIR)

    # Reset base wind farm model i.e. assumed wake model, where GP learns error between measurements and model
    model_fi = get_base_model(BASE_MODEL_FLORIS_DIR)
    
    # reset Xyk_train
    for gp in gprs:
        gp.reset_matrices()
    
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
    k_train_frames = [[[] for gp in range(len(gprs))] for i in range(kmax_gp)]
    training_size = [[0 for gp in range(len(gprs))] for i in range(kmax_gp)]
    test_var = np.nan * np.ones((kmax_gp, len(gprs)))
    # test_rmse = np.nan * np.ones((kmax_gp, len(gprs)))

    # initialize system simulator and model simulator to steady-state
    for fi in [system_fi, model_fi]:
        fi.floris.farm.flow_field.mean_wind_speed = disturbances['FreestreamWindSpeed'][0]
        fi.reinitialize_flow_field(wind_speed=disturbances['FreestreamWindSpeed'][0], wind_direction=disturbances['FreestreamWindDir'][0])
        fi.calculate_wake(yaw_angles=[disturbances[f'YawAngles_{t_idx}'][0] for t_idx in system_fi.turbine_indices],
                axial_induction=[disturbances[f'AxIndFactors_{t_idx}'][0] for t_idx in system_fi.turbine_indices])
    
    k_checkpoint = -1 # record of last time we tried to add a batch of data points to gp
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
                    # if len(gprs[gp_idx].y_train):
                    #     y_train_unique, y_train_count = np.unique(gprs[gp_idx].y_train, return_counts=True)
                    #     y_train_unique = gprs[gp_idx].y_scaler.inverse_transform(
                    #         y_train_unique.reshape((-1, 1))).reshape((1, -1))[0]
                    #     y_train_frames[-1].append(sorted(list(zip(y_train_unique, y_train_count)), key=lambda tup: tup[0]))
                    
                    # k_to_add is a list of the time-steps for which enough historic inputs exist to add to training data
                    k_to_add, effective_dk, effective_dk_to_add, history_exists \
                        = gprs[gp_idx].check_history(online_measurements_df, system_fi,
                                                     k_delay=k_delay,
                                                     dt=GP_CONSTANTS['DT'])
                    
                    # gp.k_potential is list of new data points with sufficient history
                    k_train_potential = k_to_add.loc[k_to_add > k_checkpoint].to_list()

                    # drop time-steps in online_measurements_df for which < effective_dk * k_delay
                    # compute the first k step required for the autoregressive inputs
                    # min_k_needed.update(list(
                    #     online_measurements_df['Time-Step'] - (effective_dk * GP_CONSTANTS['K_DELAY'])))
                    
                    # add new online measurements to existing set of measurements if there exists enough historic measurements
                    # unadded_k is a list of time-steps for which enough historic inputs exist to add to training data AND has not already been added to GPs training data or replay buffer
                    
                    # k_train_potential is a list of the time-steps for which enough historic inputs exist AND
                    # are not already included in k_train_replay and k_train AND
                    # do not correspond to duplicate training points
                    # Xyk_potential are new data points which have not been added to the replay buffer or training data yet and don't include any duplicates
                    # Xyk_train_replay are data points potential new data points and data points which have been dropped from training data. It is capped to a maximum size.
                    
                    if len(k_train_potential) < batch_size:
                        if COLLECT_K_TRAIN:
                            for k_tr in gprs[gp_idx].k_train:
                                k_train_frames[k_gp][gp_idx].append(k_tr)
                            training_size[k_gp][gp_idx] = len(gprs[gp_idx].k_train)
                        
                        if COLLECT_TEST_VARIANCE:
                            if gprs[gp_idx].X_test is not None:
                                test_var[k_gp, gp_idx] = gprs[gp_idx].compute_test_var()
                        
                        continue

                    
                    # remove from gp.k_potential duplicate data points within Xy_potential and Xy_replay
                    X_current, X_train_potential, y_train_potential, k_train_potential = \
                        gprs[gp_idx].find_unique_data(online_measurements_df, k_train_potential,
                                                    effective_dk,
                                                    y_modeled=[y_modeled[k][gp_idx] for k in k_train_potential],
                                                    k_delay=k_delay)
                    gprs[gp_idx].generate_X_test(X_current)
                    if len(k_train_potential) < batch_size:
                        print(
                            f'Not enough history available to fit {gp_idx} th GP with batch of samples, adding to buffer instead')
                        
                        if COLLECT_K_TRAIN:
                            for k_tr in gprs[gp_idx].k_train:
                                k_train_frames[k_gp][gp_idx].append(k_tr)
                            training_size[k_gp][gp_idx] = len(gprs[gp_idx].k_train)
                        
                        if COLLECT_TEST_VARIANCE:
                            test_var[k_gp, gp_idx] = gprs[gp_idx].compute_test_var()
                            
                        continue
                    
                    print(f'Enough history available to fit {gp_idx} th GP with {len(k_train_potential)} new samples '
                          f'between k = {k_train_potential[0]} and {k_train_potential[-1]}')

                    k_checkpoint = k_gp
                    gprs[gp_idx].choose_new_data(X_train_potential, y_train_potential, k_train_potential,
                                                 n_datapoints=batch_size)
                    
                    # add this GPs data points identifiers for this time-step
                    for k_tr in gprs[gp_idx].k_train:
                        k_train_frames[k_gp][gp_idx].append(k_tr)
                    training_size[k_gp][gp_idx] = len(gprs[gp_idx].k_train)
                    
                    # refit gp
                    if FIT_ONLINE:
                        start_time = time.time()
                        gprs[gp_idx].fit()
                        run_time = (time.time() - start_time)
                        print(f'Execution time in seconds = {str(run_time)}')
                        
                        if COLLECT_TEST_VARIANCE:
                            test_var[k_gp, gp_idx] = gprs[gp_idx].compute_test_var()
                        # test_score[k_gp, gp_idx] = gprs[gp_idx].compute_test_rmse()

                # min_k_needed = max(min(min_k_needed), 0)
                # if min_k_needed > 0:
                #     print(f'Removing samples up to k={min_k_needed} from online measurements buffer because no GP needs '
                #           f'them for autoregressive inputs.')
                #     online_measurements_df = online_measurements_df.loc[min_k_needed:]

            # for each downstream turbine
            for gp_idx, ds_idx in enumerate(system_fi.downstream_turbine_indices):
                # predict effective wind speed
                y_pred_k, y_std_k = gprs[gp_idx].predict(online_measurements_df, system_fi=system_fi,
                                                         k_delay=k_delay, y_modeled=y_modeled_k)
               
                y_pred[k_gp].append(y_pred_k)
                y_std[k_gp].append(y_std_k)
                
                y_meas_k = current_measurements[f'TurbineWindSpeeds_{ds_idx}'][0]
                y_meas[k_gp].append(y_meas_k)

    y_true = np.vstack(y_true)
    y_modeled = np.vstack(y_modeled)
    y_pred = np.vstack(y_pred)
    y_std = np.vstack(y_std)
    y_meas = np.vstack(y_meas)
    
    test_var = np.vstack(test_var) if COLLECT_TEST_VARIANCE else None
    training_size = np.vstack(training_size) if COLLECT_K_TRAIN else None

    results = {'true': y_true, 'modeled': y_modeled, 'pred': y_pred, 'std': y_std, 'meas': y_meas,
               'test_var': test_var,
               'k_train': k_train_frames, 'max_training_size': max_training_size, 'training_size': training_size,
               'mean_wind_speed': mean_wind_speed, 'mean_wind_dir': mean_wind_dir}
    
    filename = f'simulation_data_{dataset_type}_case-{case_idx}_df-{simulation_idx}'
    with open(os.path.join(SIM_SAVE_DIR, filename), 'wb') as fp:
        pickle.dump(results, fp)

    return results


if __name__ == '__main__':
    ## TURBINE WIND SPEED GP

    # Fetch wind farm system layout information, floris interface used to simulate 'true' wind farm
    system_fi = get_system_info(FLORIS_DIR)
    
    if GENERATE_PLOTS:
        system_fi.reinitialize_flow_field(wind_speed=10, wind_direction=260)
        system_fi.calculate_wake(yaw_angles=[5, 10, 5, 10, 15, 10, 15, 20, 15])
        farm_fig = plot_wind_farm(system_fi)
        farm_fig.show()
        farm_fig.savefig(os.path.join(FIG_DIR, f'wind_farm.png'))
    
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
    # wake_field_dfs = get_dfs(df_indices, TS_SAVE_DIR, proportion_training_data=GP_CONSTANTS['PROPORTION_TRAINING_DATA'])
    csv_paths = get_paths(TS_SAVE_DIR, df_indices=df_indices)
    # time, X, y, current_input_labels, input_labels \
    if PARALLEL and RUN_SIMULATIONS:
        pool = Pool(mp.cpu_count())
        case_gprs_tmp = pool.starmap(initialize,
                                     [(case_idx, system_fi,
                                       case['k_delay'], case['max_training_size'],
                                       case['kernel'])
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
                case_gprs.append(initialize(case_idx, system_fi,
                           case['k_delay'], case['max_training_size'], case['kernel']))
            else:
                case_gprs.append(None)

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
        simulation_train_cases = np.concatenate([[(case_idx, case_gprs[case_idx], df_path, df_idx,
                                                   current_input_labels,
                                                   case['k_delay'], case['noise_std'], case['batch_size'],
                                                   case['max_training_size'], 'train')
                                                  for df_idx, df_path in enumerate(csv_paths)]
                                                 for case_idx, case in enumerate(cases) if case is not None])

        pool = Pool(mp.cpu_count())
        simulations_train = pool.starmap(run_single_simulation, simulation_train_cases)
        pool.close()
    elif RUN_SIMULATIONS:
        simulations_train = []
        for case_idx, case in enumerate(cases):
            if case is not None:
                for df_idx, df_path in enumerate(csv_paths):
                    simulations_train.append(run_single_simulation(case_idx, case_gprs[case_idx], df_path, df_idx,
                                                   current_input_labels,
                                                   case['k_delay'], case['noise_std'], case['batch_size'],
                                                                   case['max_training_size'], 'train'))
                

    if GENERATE_PLOTS:
        system_fi = get_system_info(FLORIS_DIR)
        ## FETCH SIMULATION RESULTS
        def get_simulation_results(filepath, i):
            case_idx = int(filepath.split('case-')[-1].split('_')[0])
            sim_idx = int(filepath.split('df-')[-1].split('_')[0])
            with open(filepath, 'rb') as fp:
                return case_idx, sim_idx, pickle.load(fp)
            
        simulation_filepaths = []
        for root, dir, filenames in os.walk(SIM_SAVE_DIR):
            for filename in sorted(filenames):
                if 'simulation_data' in filename:
                    simulation_filepaths.append(os.path.join(SIM_SAVE_DIR, filename))
                    # case_idx = int(filename[filename.index('case') + len('case') + 1])
                    # sim_idx = int(filename[filename.index('df') + len('df') + 1])
                    # with open(os.path.join(SIM_SAVE_DIR, filename), 'rb') as fp:
                    #     simulation_results.append((case_idx, sim_idx, pickle.load(fp)))
        
        pool = mp.Pool(mp.cpu_count())
        simulation_results = pool.starmap(get_simulation_results, [(fp, 0) for fp in simulation_filepaths])
        pool.close()
        
        ## PLOT RESULTS
        # plot the true vs predicted gp values over the course of the simulation
        
        # score_type = 'rmse'
        # turbine_sim_score is n_downstream_turbines X n_simulations matrix of scores
        # for each case, simulation, turbine => RMSE, R2, median rel error, max rel error
        scores_df = compute_scores(system_fi, cases, simulation_results)
        
        
        errors_df = compute_errors(system_fi, cases, simulation_results)

        for case_idx, case in enumerate(cases):
            if case is None:
                continue
            print(f'\nCase {case_idx}')
            print(f'N_tr = {case["max_training_size"]}')
            print(f'kernel = {case["kernel"]}')
            print(f'noise_std = {case["noise_std"]}')
            print(f'k_delay = {case["k_delay"]}')
            print(f'batch_size = {case["batch_size"]}')
            case_scores = scores_df.loc[scores_df["Case"] == case_idx][['rmse', 'r2']]
            print(f'Score Mean, Median over all Simulations for averaged over Turbines '
                  f'= \n{case_scores.mean(), case_scores.median()}')
            print(f'Score Std. Dev. over all Simulations averaged over Turbines '
                  f'= \n{case_scores.std()}')
         
        scores_by_case_df = scores_df.groupby('Case')[['rmse', 'r2', 'median_rel_error', 'max_rel_error']].median().sort_values(by='rmse', ascending=True)
        scores_by_case_df['max_training_size'] = [cases[case_idx]['max_training_size'] for case_idx in scores_by_case_df.index]
        scores_by_case_df['kernel'] = [cases[case_idx]['kernel'] for case_idx in scores_by_case_df.index]
        scores_by_case_df['noise_std'] = [cases[case_idx]['noise_std'] for case_idx in scores_by_case_df.index]
        scores_by_case_df['k_delay'] = [cases[case_idx]['k_delay'] for case_idx in scores_by_case_df.index]
        scores_by_case_df['batch_size'] = [cases[case_idx]['batch_size'] for case_idx in scores_by_case_df.index]
        scores_by_case_df.to_csv(os.path.join(FIG_DIR, 'scores_by_case.csv'))
        
        generate_scores_table(scores_by_case_df, FIG_DIR)
        
        n_ts_plots = 2
        best_case_idx = scores_by_case_df.index[0]
        best_case_scores_df = scores_df.loc[scores_df['Case'] == best_case_idx]
        best_case_errors_df = errors_df.loc[errors_df['Case'] == best_case_idx]
        best_case_scores_df.to_csv(os.path.join(FIG_DIR, 'best_case_scores.csv'))
        if False:
            fn = '/Users/aoifework/Downloads/best_case_scores.csv'
            best_case_scores_df = pd.read_csv(fn, index_col=0)
            # get greatest outlying RMSE and median over all simulations for each turbine
            print('Greatest RMSE_d value over all simulations',
                  best_case_scores_df.groupby(by='Turbine').max()['rmse'].sort_values())
            print('Median RMSE_d value over all simulations',
                  best_case_scores_df.groupby(by='Turbine').median()['rmse'].sort_values())
        
        best_case_errors_df.to_csv(os.path.join(FIG_DIR, 'best_case_errors.csv'))
        if False:
            fn = '/Users/aoifework/Downloads/best_case_errors.csv'
            best_case_error_df = pd.read_csv(fn, index_col=0)
            best_case_error_df['abs_rel_error'] = best_case_error_df['rel_error'].abs()
            # get greatest outlying RMSE and median over all simulations for each turbine
            print('Greatest RelError_d value over all simulations',
                  best_case_error_df.groupby(by='Turbine').max()['abs_rel_error'].sort_values(ascending=False))
            print('Median RelError_d value over all simulations',
                  best_case_error_df.groupby(by='Turbine').median()['abs_rel_error'].sort_values(ascending=False))
        
        best_errors_df = best_case_scores_df.groupby('Turbine')[['median_rel_error', 'max_rel_error']].max().sort_values(by='median_rel_error', ascending=True)
        generate_errors_table(best_errors_df, FIG_DIR, best_case_idx)
        
        scores_case_df = best_case_scores_df.groupby('Simulation')['rmse'].median().sort_values(ascending=True)
        best_sim_indices = scores_case_df.index[:n_ts_plots]
        best_sim_indices = np.random.choice(pd.unique(best_case_scores_df['Simulation']), n_ts_plots)
        
        print('best_sim_indices', best_sim_indices)

        # print('Mean Wind Speeds for Simulations', np.round([np.mean(simulation_results[i][2]['true'][:, 0]) for i in best_sim_indices]))
        print('Simulation Freestream Wind Params')
        # print([wake_field_dfs['train'][i][['FreestreamWindSpeed', 'FreestreamWindDir']].mean() for i in best_sim_indices])
        print('Simulation Freestream Wind Speed', [sim_res[2]['mean_wind_speed'] for sim_res in simulation_results
                                                   if sim_res[1] in best_sim_indices and sim_res[0] == best_case_idx])
        print('Simulation Freestream Wind Dir', [sim_res[2]['mean_wind_dir'] for sim_res in simulation_results
                                                 if sim_res[1] in best_sim_indices and sim_res[0] == best_case_idx])

        score_fig = plot_score(system_fi, best_case_scores_df, best_case_errors_df)
        score_fig.show()
        score_fig.savefig(os.path.join(FIG_DIR, f'score.png'))
        
        if len(system_fi.floris.farm.turbines) == 9:
            ds_indices = [7, 8]#[7, 8]
        else:
            # for 2 turbine farm
            ds_indices = [1]
        
        # TODO plot ax ind factor and yaw angles of all upstream turbines for each df case
        time_ts = np.arange(0, TMAX, GP_DT)
        best_case_simulation_results = [(sim_res[1], sim_res[2]) for sim_res in simulation_results
                                        if sim_res[0] == best_case_idx and sim_res[1] in best_sim_indices]
         
        ts_fig = plot_ts(system_fi.downstream_turbine_indices, ds_indices,
                         best_case_simulation_results, time_ts, figsize=FIGSIZE)
        ts_fig.show()
        ts_fig.savefig(os.path.join(FIG_DIR, f'time_series.png'))

        full_ts_fig = plot_ts(system_fi.downstream_turbine_indices, system_fi.downstream_turbine_indices,
                         best_case_simulation_results, time_ts, figsize=(FIGSIZE[0] * len(ds_indices) / 2, FIGSIZE[1]))
        full_ts_fig.show()
        full_ts_fig.savefig(os.path.join(FIG_DIR, f'full_time_series.png'))
       
        if COLLECT_TEST_VARIANCE:
            std_fig = plot_std_evolution(system_fi.downstream_turbine_indices, ds_indices,
                                         best_case_simulation_results, time_ts)
            std_fig.show()
            std_fig.savefig(os.path.join(FIG_DIR, f'std_evolution.png'))
        
        if COLLECT_K_TRAIN:
            k_train_fig = plot_k_train_evolution(system_fi.downstream_turbine_indices, ds_indices,
                                                 best_case_simulation_results, time_ts)
            k_train_fig.show()
            k_train_fig.savefig(os.path.join(FIG_DIR, f'k_train_evolution.png'))
        
        # PLOT_MULT_TURBINES = True
        # error_ds_indices = [[3,4,5], [6,7,8]] # grouped for each axis
        # error_ts_fig = plot_error_ts(system_fi.downstream_turbine_indices,
        #                              error_ds_indices,
        #                              best_case_simulation_results, time_ts)
        # error_ts_fig.show()
        # error_ts_fig.savefig(os.path.join(FIG_DIR, f'error_ts.png'))

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