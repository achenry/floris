from system import System
from mpc import MPC
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import norm
from DownstreamTurbineGPR import GP_CONSTANTS, get_system_info, get_data, init_gprs, SAVE_DIR, FIG_DIR, get_base_model, get_dfs
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib as mpl
from preprocessing import add_gaussian_noise
import multiprocessing as mp
from multiprocessing import Pool
from plotting import plot_training_data
import matplotlib.animation as animation
from matplotlib.animation import FuncAnimation, FFMpegWriter

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

# TODO results
#  1) train offline, run simulations for training and testing datasets,
#     plot true vs predicted values of downstream TurbineWindSpeeds

# TODO
#  1) offline learning for same training data and base models - value model type X
#  2) offline learning for same training data and base models - error model type X
#  3) offline learning for different training data and base models for learning error X
#  4) offline learning with noisy measurements (NOISE_STD != 0)
#  5) Test Online data selection algorithm (choose highest std, drop randomly and plot cluster of evolving data points X
#  6) Test variable wind speed in generate_wake_field.py (WS_TI != 0) OR just assume you've filtered it
#  7) online learning with noisy data (NOISE_STD != 0)
#  8) Test for different kernel functions, values of MAX_TRAINING_SIZE, NOISE_STD, k_delay, batch size
#  9) Test for 9 x 9 Wind Farm layout
#  10) keep bank of historic measurements that have not been added to training data and randomly select from this, or select most optimal for reducing overall variance

# TODO run yaw/ax ind factor controllers with GP estimate vs model
# TODO RMSE mean vs different kernel functions, or exploration vs. exploitation parameter

RUN_MPC = False
RUN_GP = True
PLOT_GP = True
TMAX = 600
TRAIN_ONLINE = True
FIT_ONLINE = True

# construct case hierarchy
MAX_TRAINING_SIZE_VALS = [25, 50, 100, 200, 400]
default_max_training_size = 100
NOISE_STD_VALS = [0.001, 0.01, 0.1]
default_noise_std = 0.01
K_DELAY_VALS = [2, 4, 6, 8, 10]
default_k_delay = 6
BATCH_SIZE_VALS = [1, 5, 10, 25, 50]
default_batch_size = 10

cases = [{'max_training_size': default_max_training_size, 'noise_std': default_noise_std,
          'k_delay': default_k_delay, 'batch_size': x} for x in BATCH_SIZE_VALS] + \
        [{'max_training_size': default_max_training_size, 'noise_std': default_noise_std,
          'k_delay': x, 'batch_size': default_batch_size} for x in K_DELAY_VALS] + \
        [{'max_training_size': default_max_training_size, 'noise_std': x,
          'k_delay': default_k_delay, 'batch_size': default_batch_size} for x in NOISE_STD_VALS] + \
        [{'max_training_size': x, 'noise_std': default_noise_std,
          'k_delay': default_k_delay, 'batch_size': default_batch_size} for x in MAX_TRAINING_SIZE_VALS]

# max_training_size = MAX_TRAINING_SIZE_VALS[0]
# noise_std = NOISE_STD_VALS[0]
# k_delay = K_DELAY_VALS[0]
# batch_size = BATCH_SIZE_VALS[0]

if not os.path.exists(os.path.join(SAVE_DIR)):
    os.makedirs(SAVE_DIR)

if not os.path.exists(os.path.join(FIG_DIR)):
    os.makedirs(FIG_DIR)

def initialize(full_offline_measurements_df, system_fi, X, y, input_labels, k_delay, noise_std, max_training_size):
    # Define normalization procedure
    X_scalar = StandardScaler().fit(X['train'])
    y_scalar = StandardScaler().fit(y['train'])

    gprs = init_gprs(system_fi, X_scalar, y_scalar, input_labels)

    # add noise to Turbine Wind Speed measurements
    noisy_measurements_df = add_gaussian_noise(system_fi, full_offline_measurements_df, std=noise_std)

    for gp in gprs:

        gp.max_training_size = max_training_size

        k_to_add, effective_dk, reduced_effective_dk, history_exists = \
            gp.check_history(noisy_measurements_df, system_fi, k_delay=k_delay, dt=GP_CONSTANTS['DT'])

        if not history_exists:
            continue

        # shuffle order to give random data points to GP, then truncate to number of data points needed
        shuffle_idx = list(k_to_add.index)
        np.random.shuffle(shuffle_idx)
        shuffle_idx = shuffle_idx[:max_training_size]

        # compute the base model values for the wake, given all of the freestream wind speeds, yaw angles and axial induction factors over time for each time-series dataset
        y_modeled = []
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
                model_fi.floris.farm.flow_field.mean_wind_speed = noisy_measurements_df.loc[
                    k, 'FreestreamWindSpeed']
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
                y_modeled.append(model_fi.floris.farm.turbines[gp.turbine_index].average_velocity)
            y_modeled = np.array(y_modeled)
            noisy_measurements_df[f'TurbineWindSpeeds_{gp.turbine_index}'].to_numpy()
        else:
            y_modeled = np.zeros(len(noisy_measurements_df.index))

        X_train_new, y_train_new = gp.prepare_data(noisy_measurements_df, k_to_add.loc[shuffle_idx],
                                                   reduced_effective_dk.loc[shuffle_idx],
                                                   y_modeled=y_modeled[shuffle_idx],
                                                   k_delay=k_delay)

        gp.add_data(X_train_new, y_train_new, k_to_add.loc[shuffle_idx], is_online=False)

        # Fit data
        gp.fit()

    return gprs, X_scalar, y_scalar

def run_single_simulation(case_idx, gprs, simulation_df, simulation_idx, X_scalar, y_scalar,
                          current_input_labels, k_delay, noise_std, batch_size):

    # Fetch wind farm system layout information, floris interface used to simulate 'true' wind farm
    system_fi = get_system_info()

    # Reset base wind farm model i.e. assumed wake model, where GP learns error between measurements and model
    model_fi = get_base_model()

    print('Defining wind, yaw, ax ind factor disturbances')

    disturbances = {
        'FreestreamWindSpeed': simulation_df['FreestreamWindSpeed'],
        'FreestreamWindDir': simulation_df['FreestreamWindDir']
    }
    for t_idx in system_fi.turbine_indices:
        disturbances[f'YawAngles_{t_idx}'] = simulation_df[f'YawAngles_{t_idx}']
        disturbances[f'AxIndFactors_{t_idx}'] = simulation_df[f'AxIndFactors_{t_idx}']

    online_measurements_df = pd.DataFrame(columns=current_input_labels) # holds all current and historic time-step data needed to make autoregressive inputs
    k_buffered = [] # list of discrete time-steps for which the data point has not been added to the training data yet for lack of history for the autoregressive inputs

    y_true = [[] for i in range(int(TMAX // GP_DT))]
    y_modeled = [[] for i in range(int(TMAX // GP_DT))]
    y_pred = [[] for i in range(int(TMAX // GP_DT))]
    y_std = [[] for i in range(int(TMAX // GP_DT))]
    y_train_frames = []

    # initialize system simulator and model simulator to steady-state
    for fi in [system_fi, model_fi]:
        fi.floris.farm.flow_field.mean_wind_speed = disturbances['FreestreamWindSpeed'][0]
        fi.reinitialize_flow_field(wind_speed=disturbances['FreestreamWindSpeed'][0], wind_direction=disturbances['FreestreamWindDir'][0])
        fi.calculate_wake(yaw_angles=[disturbances[f'YawAngles_{t_idx}'][0] for t_idx in system_fi.turbine_indices],
                axial_induction=[disturbances[f'AxIndFactors_{t_idx}'][0] for t_idx in system_fi.turbine_indices])

    for k_sys, t in enumerate(range(0, TMAX, SYS_DT)):
        print(f'Simulating {k_sys}th Time-Step')
        # plot every 100 seconds
        # plot_data = True if t % 100 == 0 else False

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

        # if GP sampling time has passed AND predicted std at sample exceeds a given threshold
        if t % GP_DT == 0:
            k_gp = int(t // GP_DT)

            # for each downstream turbine
            for gp_idx, ds_idx in enumerate(system_fi.downstream_turbine_indices):
                y_modeled_k = model_fi.floris.farm.turbines[ds_idx].average_velocity if GP_CONSTANTS[
                                                                                            'MODEL_TYPE'] == 'error' else 0
                y_true_k = system_fi.floris.farm.turbines[ds_idx].average_velocity
                y_true[k_gp].append(y_true_k)
                y_modeled[k_gp].append(y_modeled_k)

            print(f'Adding {k_gp}th GP Sample')

            # collect current measurements in buffer: time
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
                # collect new measurements: effective wind speed at downstream turbine (target)
                current_measurements[f'TurbineWindSpeeds_{ds_turbine_idx}'].append(
                    system_fi.floris.farm.turbines[ds_turbine_idx].average_velocity)

            # add collected data to online training buffer,
            # only include inputs which consider turbines upstream of this turbine (based on radius)
            current_measurements = pd.DataFrame(add_gaussian_noise(system_fi, current_measurements, std=noise_std))
            online_measurements_df = pd.concat([online_measurements_df, current_measurements],
                                               ignore_index=True)
            # to_add_measurements = pd.concat([to_add_measurements, current_measurements], ignore_index=True)
            k_buffered.append(k_gp)
            assert online_measurements_df['Time'].iloc[-1] == k_gp
            assert (np.diff(online_measurements_df.index) == 1).all()

            # if X times enough samples have been added to to_add_measurements to make a batch
            if len(k_buffered) >= 2 * batch_size and TRAIN_ONLINE:

                # drop the historic inputs we no longer need for ANY downstream turbine
                # i.e. keep the last max(effective_dk * k_delay) for all downstream turbines
                # IF we used some of the current measurements
                min_k_needed = set()
                buffered_measurements = online_measurements_df.loc[
                                                         [(t in k_buffered * GP_DT) for t in online_measurements_df['Time']]]

                y_train_frames.append([])

                # for each downstream turbine
                for gp_idx, ds_turbine_idx in enumerate(system_fi.downstream_turbine_indices):

                    # plot evolution of gprs[gp_idx].y_train
                    y_train_unique, y_train_count = np.unique(gprs[gp_idx].y_train, return_counts=True)
                    y_train_frames[-1].append(sorted(list(zip(y_scalar.inverse_transform(
                        y_train_unique.reshape((-1, 1))).reshape((1, -1))[0], y_train_count)), key=lambda tup: tup[0]))
                    # x = y_scalar.inverse_transform(y_train_unique.reshape((-1, 1))).squeeze()
                    k_to_add, effective_dk, reduced_effective_dk, history_exists \
                        = gprs[gp_idx].check_history(buffered_measurements, system_fi,
                                                     k_delay=k_delay,
                                                     dt=GP_CONSTANTS['DT'])

                    # drop time-steps in k_buffered which < effective_dk * k_delay
                    # compute the first k step required for the autoregressive inputs
                    min_k_needed.update(list(
                        k_buffered - (effective_dk * GP_CONSTANTS['K_DELAY'])))

                    # if not history_exists:
                    if len(k_to_add) < batch_size:
                        print(f'Not enough history available to fit {gp_idx} th GP with batch of samples, adding to buffer instead')
                        continue

                    print(f'Enough history available to fit {gp_idx} th GP with {k_to_add} th samples')
                    for k in k_to_add:
                        del k_buffered[k_buffered.index(k)]

                    # add new online measurements to existing set of measurements if there exists enough historic measurements
                    potential_X_train_new, potential_y_train_new \
                        = gprs[gp_idx].prepare_data(online_measurements_df, k_to_add, effective_dk,
                                                    y_modeled=[y_modeled[k][gp_idx] for k in k_to_add.to_numpy()],
                                                    k_delay=k_delay)

                    gprs[gp_idx].choose_new_data(potential_X_train_new, potential_y_train_new, k_to_add)

                    # refit gp
                    if FIT_ONLINE:
                        gprs[gp_idx].fit()

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

        if t % MPC_DT == 0:
            pass
            # read current state x

            # get distrubance predictions over horizon

            # warm-start mpc solution with most recent solution

            # run mpc

            # get solution u*

        # apply most recent solution to u* to system

    y_true = np.vstack(y_true)
    y_modeled = np.vstack(y_modeled)
    y_pred = np.vstack(y_pred)
    y_std = np.vstack(y_std)

    results = {'true': y_true, 'modeled': y_modeled, 'pred': y_pred, 'std': y_std}

    training_fig, training_ax = plt.subplots(facecolor = plt.cm.Greys(0.2),
                      dpi = 150,
                      tight_layout=True)
    training_max_count = max(max(tup[1] for tup in arr[gp_idx]) for gp_idx in range(len(gprs)) for arr in y_train_frames)
    training_y_vals = sorted(set(np.concatenate([[tup[0] for tup in arr[gp_idx]]
                                               for gp_idx in range(len(gprs)) for arr in y_train_frames])))

    def animate(k_idx, *fargs):
        # training_ax = training_fig.add_subplot(1, 1, 1)
        training_ax.clear()
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
    gp_idx = 0
    ani = FuncAnimation(training_fig, animate, frames=len(y_train_frames), interval=25, fargs=[gp_idx])
    ani.save(f'./figs/training_ani_case-{case_idx}_df-{simulation_idx}_gp.mp4')

    with open(os.path.join(SAVE_DIR, f'simulation_data_case-{case_idx}_df-{simulation_idx}'), 'wb') as fp:
        pickle.dump(results, fp)


if __name__ == '__main__':
    ## MPC
    if RUN_MPC:
        # Define System
        sys = System(2, 1, 0, 1, 'forward_difference', dt=0.1, 
                    linear_matrices={'A': [[0, 1], [0, 0]],
                                    'Bu': [[0], [1]],
                                    'Bd': [[0], [1]],
                                    'C': [[1, 0]],
                                    'Du': [[0]],
                                    'Dd': [[]]}, 
                    is_discrete=False)
        
        # Test Open-Loop Simulation
        u_traj = np.sin(np.arange(0, 10 * np.pi, 0.1))[:, np.newaxis]
        d_traj = [[] for i in range(len(u_traj))]
        x0 = [0, -1]
        # sys.simulate(x0, u_traj, d_traj)
        
        # Define Cost Function
        Q = np.array([[1, 0], [0, 1]])
        P = Q
        R = np.array([[1]])
        stage_cost_func = lambda xj, uj: xj @ Q @ xj + uj @ R @ uj
        term_cost_func = lambda xN: xN @ P @ xN
        stage_cost_jac_func = lambda xj, uj: np.concatenate([2 * Q @ xj, 2 * R @ uj])
        term_cost_jac_func = lambda xN: np.array(2 * P @ xN)
        stage_cost_hess_func = lambda xj, uj: block_diag(2 * Q, 2 * R)
        term_cost_hess_func = lambda xN: np.array(2 * P)
        print(stage_cost_func(x0, u_traj[0, :])) # expect
        print(term_cost_func(x0))
        print(stage_cost_jac_func(x0, u_traj[0, :]))
        print(stage_cost_hess_func(x0, u_traj[0, :]))
        print(term_cost_jac_func(x0))
        print(term_cost_hess_func(x0))
        
        # Define Constraints
        # TODO test adding one at a time
        stage_ineq_constraint_func = lambda xj, uj: [] # [xj[0] - 10, -xj[0] + 10] #, uj[0] - 20, -uj[0] + 20]
        term_ineq_constraint_func = lambda xN: [] # [xN[0] - 10, -xN[0] + 10]
        stage_ineq_constraint_jac_func = lambda xj, uj: np.zeros((0, sys.n_states + sys.n_ctrl_inputs)) # np.vstack([[1, 0, 0], [-1, 0, 0]])#, [0, 0, 1], [0, 0, -1]])
        term_ineq_constraint_jac_func = lambda xN: np.zeros((0, sys.n_states)) # np.vstack([[1, 0], [-1, 0]])
        print(stage_ineq_constraint_func(x0, u_traj[0, :]))
        print(term_ineq_constraint_func(x0))
        print(stage_ineq_constraint_jac_func(x0, u_traj[0, :]))
        print(term_ineq_constraint_jac_func(x0))
        
        # Define normal disturbance distribution
        d_pdf_traj = [[] for i in range(len(u_traj))]
        norm.pdf()
        
        # Define MPC problem
        horizon_length = 10
        dt_control = 0.1
        mpc = MPC(sys, 
                stage_cost_func, term_cost_func, 
                stage_cost_jac_func, term_cost_jac_func,
                stage_cost_hess_func, term_cost_hess_func,
                stage_ineq_constraint_func, term_ineq_constraint_func, 
                stage_ineq_constraint_jac_func, term_ineq_constraint_jac_func,
                horizon_length,
                dt_control)
        
        opt_vars = np.zeros(mpc.n_opt_vars)
        print(mpc.cost_func(opt_vars))
        print(mpc.cost_jac_func(opt_vars))
        print(mpc.cost_hess_func(opt_vars))
        print(mpc.ineq_constraint_func(opt_vars))
        print(mpc.ineq_constraint_jac_func(opt_vars), np.array(mpc.ineq_constraint_jac_func(opt_vars)).shape)
        
        mpc.simulate(x0, d_traj, t_max=(len(d_traj) - horizon_length) * sys.dt)
        
        pass
    
    if RUN_GP:
        ## TURBINE WIND SPEED GP

        # Fetch wind farm system layout information, floris interface used to simulate 'true' wind farm
        system_fi = get_system_info()

        # Fetch base wind farm model i.e. assumed wake model, where GP learns error between measurements and model
        model_fi = get_base_model()
        
        # Read raw offline training data
        print('Reading offline training data')

        # consider the following time-series cases
        df_indices = None #list(range(20))s
        wake_field_dfs = get_dfs(df_indices, proportion_training_data=GP_CONSTANTS['PROPORTION_TRAINING_DATA'])
        full_offline_measurements_df = pd.concat(wake_field_dfs['train'], ignore_index=True)

        # time, X, y, current_input_labels, input_labels \
        case_data = {k_delay: get_data(wake_field_dfs, system_fi,
                       model_type=GP_CONSTANTS['MODEL_TYPE'], k_delay=k_delay, dt=GP_CONSTANTS['DT'],
                       model_fi=model_fi, collect_raw_data_bool=GP_CONSTANTS['COLLECT_RAW_DATA'],
                       plot_data_bool=GP_CONSTANTS['PLOT_DATA']) for k_delay in K_DELAY_VALS}

        pool = Pool(mp.cpu_count())
        case_gprs = pool.starmap(initialize,
                     [(full_offline_measurements_df, system_fi,
                       case_data[case['k_delay']][1], case_data[case['k_delay']][2], case_data[case['k_delay']][4],
                       case['k_delay'], case['noise_std'], case['max_training_size'])
                      for case in cases])
        pool.close()

        # start simulation
        GP_DT = GP_CONSTANTS['DT']
        SYS_DT = 1
        MPC_DT = 5
        K_SYS_MAX = int(TMAX // SYS_DT)
        time_steps = range(K_SYS_MAX)

        for case_idx, case in enumerate(cases):
            # Run simulations
            pool = Pool(mp.cpu_count())
            simulations_train = pool.starmap(run_single_simulation,
                                             [(case_idx, case_gprs[case_idx][0], df, df_idx,
                                               case_gprs[case_idx][1], case_gprs[case_idx][2],
                                               case_data[case['k_delay']][3],
                                               case['k_delay'], case['noise_std'], case['batch_size'])
                                              for df_idx, df in enumerate(wake_field_dfs['train'])])
            simulations_test = pool.starmap(run_single_simulation,
                                             [(case_idx, case_gprs[case_idx][0], df, df_idx + len(wake_field_dfs['train']),
                                               case_gprs[case_idx][1], case_gprs[case_idx][2],
                                               case_data[case['k_delay']][3],
                                               case['k_delay'], case['noise_std'], case['batch_size'])
                                              for df_idx, df in enumerate(wake_field_dfs['test'])])
            pool.close()

    if PLOT_GP:
        simulation_results = []
        for root, dir, filenames in os.walk(SAVE_DIR):
            for filename in sorted(filenames):
                if 'simulation_data' in filename:
                    with open(os.path.join(SAVE_DIR, filename), 'rb') as fp:
                        simulation_results.append(pickle.load(fp))

        ## PLOT RESULTS
        # plot the true vs predicted gp values over the course of the simulation

        # GP estimate, true value, noisy measurements of
        # effective wind speeds of downstream turbines vs.
        # time for one dataset

        ds_idx = 0
        time = np.arange(TMAX)
        for sim_idx in range(len(simulation_results)):
            ts_fig, ts_ax = plt.subplots(1, 1)
            ts_ax.plot(time, simulation_results[sim_idx]['true'][:, ds_idx], label='True')
            ts_ax.plot(time, simulation_results[sim_idx]['pred'][:, ds_idx], label='Predicted Mean')
            ts_ax.plot(time, simulation_results[sim_idx]['modeled'][:, ds_idx], label='Base Modeled')
            ts_ax.scatter(time, wake_field_dfs['train'][sim_idx][
                f'TurbineWindSpeeds_{system_fi.downstream_turbine_indices[ds_idx]}'], c='r', label='Measurements')
            # ts_ax_scatter(time, gprs[0].y_train, c='p', label='Training Outputs')
            # ts_ax.fill_between(time, simulation_results[sim_idx]['pred'][:, ds_idx]
            #                    - simulation_results[sim_idx]['std'][:, ds_idx],
            #                          simulation_results[sim_idx]['pred'][:, ds_idx]
            #                    + simulation_results[sim_idx]['std'][:, ds_idx],
            #                    alpha=0.1, label='Predicted Std. Dev.')
            ts_ax.legend(loc='center left')
            ts_ax.set(xlabel='Time [s]', title=f'Downstream Turbine {ds_idx} Effective Wind Speed [m/s]')
            ts_fig.show()

        plt.savefig(os.path.join(FIG_DIR, f'time_series.png'))

        # RMSE mean and std (true turbine effective wind speed vs. GP estimate) over 500 simulations for
        # each downstream turbine
        err_fig, err_ax = plt.subplots(1, 1)
        rmse_mean = []
        rmse_std = []
        # for each downstream turbine
        for i, ds_idx in enumerate(system_fi.downstream_turbine_indices):
            # compute the rmse of the turbine effective wind speed error for each simulation
            sim_rmse = [np.sqrt(np.square(np.subtract(sim['true'][:, i], sim['pred'][:, i])).mean())
                        for sim in simulation_results]
            rmse_mean.append(np.mean(sim_rmse))
            rmse_std.append(np.std(sim_rmse))

        err_ax.errorbar(x=system_fi.downstream_turbine_indices, y=rmse_mean, yerr=rmse_std)
        err_ax.set(xlabel='Downstream Turbine Index', title=f'Downstream Turbine Effective Wind Speed RMSE over all Simulations')
        err_fig.show()