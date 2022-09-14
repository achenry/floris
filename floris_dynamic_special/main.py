from system import System
from mpc import MPC
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import norm
from DownstreamTurbineGPR import GP_CONSTANTS, get_system_info, get_data, init_gprs
from generate_wake_field import step_change
import pandas as pd
from sklearn.preprocessing import StandardScaler
from collections import defaultdict

# TODO
#  1) test GP online learning with open-loop axial induction factors
#  2) use GP models to fetch mean and standard deviation over prediction horizon for each mpc step
#  3) test double-integrator mpc
#  

if __name__ == '__main__':
    
    ## MPC
    if False:
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
    
    if True: 
        ## TURBINE WIND SPEED GP
        # Fetch wind farm system layout information, floris interface used to simulate 'true' wind farm
        system_fi = get_system_info()
        
        # Read raw offline training data
        print('Reading offline training data')
        time, X, y, current_input_labels, input_labels, offline_measurements_dfs = get_data(system_fi)
        
        # Define normalization procedure
        print('Defining data normalization')
        X_scalar = StandardScaler().fit(X['full'])
        y_scalar = StandardScaler().fit(y['full'])
        
        # Initialize gp for each downstream turbine
        print('Initializing Downstream Turbine GPs')
        gprs = init_gprs(system_fi, X_scalar, y_scalar, input_labels, max_training_size=X['train'].shape[0])
        
        # Fit each gp with offline data
        print('Fitting Downstream Turbine GPs with offline training data')
        for gp in gprs:
            
            # Add offline data (normalizing first)
            # for df in offline_measurements_dfs:
            gp.add_data(pd.concat([df.loc[df['DatasetType'] == 'train'] for df in offline_measurements_dfs], ignore_index=True), system_fi)
            
            # Fit data
            gp.fit()
            
        # start simulation
        TMAX = 600
        GP_DT = GP_CONSTANTS['DT']
        SYS_DT = 1
        MPC_DT = 5
        K_SYS_MAX = int(TMAX // SYS_DT)
        time_steps = range(K_SYS_MAX)
        
        # generate disturbances: freestream wind speed, freestream wind dir, upstream turbine yaw angles
        # TODO generate training data with varying yaw angles
        # TODO add wind speed, yaw angle disturbances during online training
        print('Defining wind, yaw, ax ind factor disturbances')
        ws_mean = step_change([8, 10, 12], TMAX, SYS_DT)
        ws_ti = np.zeros(K_SYS_MAX)
        ws_dev = (ws_ti / 100) * ws_mean
        wd_mean = step_change([270], TMAX, SYS_DT)
        wd_ti = np.zeros(K_SYS_MAX)
        wd_dev = (wd_ti / 100) * wd_mean
        
        disturbances = {
            'FreestreamWindSpeed': [np.random.normal(loc=ws_mean[k], scale=ws_dev[k])[0] for k in time_steps],
            'FreestreamWindDir': [np.random.normal(loc=wd_mean[k], scale=wd_dev[k])[0] for k in time_steps]
        }
        for t_idx in system_fi.upstream_turbine_indices:
            disturbances[f'YawAngles_{t_idx}'] = np.zeros(K_SYS_MAX)
            disturbances[f'AxIndFactors_{t_idx}'] = step_change([0.11, 0.22, 0.33], TMAX, SYS_DT)
        
        online_measurements_df = pd.DataFrame(columns=current_input_labels + ['IsAdded'])
        
        # static initialization
        system_fi.floris.farm.flow_field.mean_wind_speed = ws_mean[0, 0]
        system_fi.reinitialize_flow_field(wind_speed=disturbances['FreestreamWindSpeed'][0], wind_direction=disturbances['FreestreamWindDir'][0]) 
        system_fi.calculate_wake(yaw_angles=[disturbances[f'YawAngles_{t_idx}'][0] for t_idx in system_fi.upstream_turbine_indices], 
                                axial_induction=[disturbances[f'AxIndFactors_{t_idx}'][0] for t_idx in system_fi.upstream_turbine_indices])
        
        current_measurement = defaultdict(list) 
        to_add_measurements = defaultdict(list)
        
        y_true = [[] for i in range(int(TMAX // GP_DT))]
        y_pred = [[] for i in range(int(TMAX // GP_DT))]
        y_std = [[] for i in range(int(TMAX // GP_DT))]
        
        for k_sys, t in enumerate(range(0, TMAX, SYS_DT)):
            print(f'Simulating {k_sys}th Time-Step')
            # plot every 100 seconds
            # plot_data = True if t % 100 == 0 else False
            
            # dynamic reinitialization    
            system_fi.floris.farm.flow_field.mean_wind_speed = ws_mean[k_sys, 0]
            system_fi.reinitialize_flow_field(wind_speed=disturbances['FreestreamWindSpeed'][k_sys], wind_direction=disturbances['FreestreamWindDir'][k_sys], sim_time=t)
            system_fi.calculate_wake(yaw_angles=[disturbances[f'YawAngles_{t_idx}'][k_sys] for t_idx in system_fi.upstream_turbine_indices], 
                                     axial_induction=[disturbances[f'AxIndFactors_{t_idx}'][k_sys] for t_idx in system_fi.upstream_turbine_indices], sim_time=t)
            
            # if GP sampling time has passed
            if t % GP_DT == 0:
                k_gp = int(t // GP_DT)
                print(f'Adding {k_gp}th GP Sample')
               
                # collect measurements: time 
                to_add_measurements['Time'].append(t)
                
                # collect measurements: freestream wind speed and direction
                to_add_measurements['FreestreamWindSpeed'].append(disturbances['FreestreamWindSpeed'][k_sys])
                to_add_measurements['FreestreamWindDir'].append(disturbances['FreestreamWindDir'][k_sys])
                
                # for each upstream turbine
                for us_turbine_idx in system_fi.upstream_turbine_indices:
                    # collect new measurements: axial induction factor, yaw angle (inputs)
                    to_add_measurements[f'YawAngles_{us_turbine_idx}'].append(system_fi.floris.farm.turbines[us_turbine_idx].yaw_angle)
                    to_add_measurements[f'AxIndFactors_{us_turbine_idx}'].append(system_fi.floris.farm.turbines[us_turbine_idx].ai_set[0])
                
                # for each downstream turbine
                for gp_idx, ds_turbine_idx in enumerate(system_fi.downstream_turbine_indices):
                    
                    # collect new measurements: effective wind speed at downstream turbine (target)
                    to_add_measurements[f'TurbineWindSpeeds_{ds_turbine_idx}'].append(system_fi.floris.farm.turbines[ds_turbine_idx].average_velocity)
                    
                # add collected data to online training dataset, only include inputs which consider turbines upstream of this turbine (based on radius)
                online_measurements_df = pd.concat([online_measurements_df, pd.DataFrame(to_add_measurements).iloc[-1:]], ignore_index=True)
                assert online_measurements_df.index[-1] == k_gp
                assert (np.diff(online_measurements_df.index) == 1).all()
                
                if (k_gp + 1) % GP_CONSTANTS['BATCH_SIZE'] == 0:
                    print(f'Adding {[(t // GP_DT) for t in to_add_measurements["Time"]]} th samples to training data')
                    assert len(to_add_measurements['Time']) == GP_CONSTANTS['BATCH_SIZE']
                    # for each downstream turbine
                    for gp_idx, ds_turbine_idx in enumerate(system_fi.downstream_turbine_indices):
                        
                        # add and normalize new training data
                        gprs[gp_idx].add_data(online_measurements_df, system_fi)
                        
                        # refit gp 
                        gprs[gp_idx].fit()
                    
                
                    # drop the historic inputs we no longer need for ANY downstream turbine i.e. keep the last max(effective_dk * k_delay) for all downstream turbines IF we used some of the current measurements
                    min_k_needed = []
                    
                    # for each downstream turbine
                    to_add_measurements = pd.DataFrame(to_add_measurements)
                    for gp in gprs:
                        # for each of the current measurements
                        dks = gp.compute_effective_dk(system_fi, to_add_measurements, k_delay=GP_CONSTANTS['K_DELAY'], dt=GP_DT)
                        # compute the first k step required for the autoregressive inputs
                        min_k_needed = min_k_needed + list((to_add_measurements['Time'] // GP_DT) - (dks * GP_CONSTANTS['K_DELAY']))
                        
                        assert gp.X_train.shape[0] <= gp.max_training_size
                        
                    online_measurements_df = online_measurements_df.loc[min(min_k_needed):]
                
                    to_add_measurements = defaultdict(list)
                
                # for each downstream turbine
                for gp_idx, ds_turbine_idx in enumerate(system_fi.downstream_turbine_indices): 
                    # predict effective wind speed
                    y_true_k, y_pred_k, y_std_k = gprs[gp_idx].predict(online_measurements_df, system_fi)
                    
                    y_true[k_gp].append(y_true_k)
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
        y_pred = np.vstack(y_pred)
        y_std = np.vstack(y_std)
    
        ## PLOT RESULTS
        # plot the true vs predicted gp values over the course of the simulation