# %matplotlib inline
'''
Generate 'true' wake field data for use in GP learning procedure
Inputs: Yaw Angles, Freestream Wind Velocity, Freestream Wind Direction, Turbine Topology
Need csv containing 'true' wake characteristics at each turbine (variables) at each time-step (rows).
'''

# git add . & git commit -m "updates" & git push origin
# ssh ahenry@eagle.hpc.nrel.gov
# cd ...
# sbatch ...

# from defusedxml import DTDForbidden
import matplotlib.pyplot as plt
import numpy as np
from floridyn import tools as wfct # Incoming con
import pandas as pd
import os
import sys
from multiprocessing import Pool
from CaseGen_General import CaseGen_General

if sys.platform == 'darwin':
    FARM_LAYOUT = '9turb'
    save_dir = f'./{FARM_LAYOUT}_wake_field_cases'
    data_dir = './data'
    fig_dir = './figs'
    TOTAL_TIME = 600  # five minutes
elif sys.platform == 'linux':
    FARM_LAYOUT = '9turb'
    save_dir = f'/scratch/alpine/aohe7145/wake_gp/{FARM_LAYOUT}_wake_field_cases'
    data_dir = f'/scratch/alpine/aohe7145/wake_gp/data'
    fig_dir = f'/scratch/alpine/aohe7145/wake_gp/figs'
    TOTAL_TIME = 600  # ten minutes



# ********** #
def step_change(vals, T, dt):
    step_vals = []
    k_change = int(T // dt // len(vals))
    for val in vals:
        for k in range(k_change):
            step_vals.append(val)
    
    while len(step_vals) < T // dt:
        step_vals.append(vals[-1])

    # return step_vals
    return np.vstack(step_vals)

# **************************************** Parameters **************************************** #

# total simulation time
DT = 1.0 # DOESN'T WORK WELL WITH OTHER DT VALUES
DEFAULT_AX_IND_FACTOR = 1 / 3
DEFAULT_YAW_ANGLE = 0
N_SEEDS = 1
WS_TI = 0
WD_TI = 0
DEBUG = len(sys.argv) > 1 and sys.argv[1] == 'debug'
print('debug', DEBUG)

# hold constant over simulation time span
FREESTREAM_WIND_SPEEDS = [step_change([val], TOTAL_TIME, DT) for val in ([8] if DEBUG else [8, 10, 12])]
FREESTREAM_WIND_DIRS = [step_change([val], TOTAL_TIME, DT) for val in ([260] if DEBUG else [250, 260, 270])]

# change over course of single simulation to learn how wake fields propagate over time
# TODO add cases with constant yaw angle, ax ind factor, incur changes at different times by changing number vals
YAW_ANGLES = [step_change([0.0, 7.5, 15], TOTAL_TIME, DT)] if DEBUG else \
    [step_change([-20, -15, -10], TOTAL_TIME, DT),
    step_change([-10, -15, -20], TOTAL_TIME, DT),
    step_change([20, 15, 10], TOTAL_TIME, DT),
    step_change([10, 15, 20], TOTAL_TIME, DT),
     step_change([-10, 0, 10], TOTAL_TIME, DT),
     step_change([10, 0, -10], TOTAL_TIME, DT),
     step_change([10], TOTAL_TIME, DT),
     step_change([-10], TOTAL_TIME, DT),
     step_change([0], TOTAL_TIME, DT)]
AX_IND_FACTORS = [step_change([0.11, 0.22, 0.22, 0.33], TOTAL_TIME, DT)] if DEBUG else \
    [step_change([0.11, 0.22, 0.22, 0.33], TOTAL_TIME, DT),
     step_change([0.33, 0.33, 0.22, 0.11], TOTAL_TIME, DT),
     step_change([0.22, 0.22, 0.33, 0.11], TOTAL_TIME, DT),
     step_change([0.11], TOTAL_TIME, DT),
     step_change([0.33], TOTAL_TIME, DT),
     step_change([0.22], TOTAL_TIME, DT)]

# **************************************** Initialization **************************************** #
# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
# floris_dir = "./2turb_floris_input.json"

floris_dir = f"./{FARM_LAYOUT}_floris_input.json"

# Initialize
fi = wfct.floris_interface.FlorisInterface(floris_dir)
fi.reinitialize_flow_field(wind_speed=8, wind_direction=270) 
fi.calculate_wake()
n_turbines = len(fi.floris.farm.turbines)

# Reinitialize
# start_ws = 8
# start_wd = 250.
# fi.reinitialize_flow_field(wind_speed=start_ws, wind_direction=start_wd)

# make case save dir
# for dir in [save_dir, data_dir, fig_dir]:
#     if not os.path.exists(dir):
#         os.makedirs(dir)

# **************************************** GENERATE TIME-VARYING FREESTREAM WIND SPEED/DIRECTION, YAW ANGLE, TURBINE TOPOLOGY SWEEP **************************************** #
# TODO - alt generate DLCs using turbsim

case_inputs = {}
case_inputs['mean_wind_speed'] = {'group': 0, 'vals': FREESTREAM_WIND_SPEEDS}
case_inputs['mean_wind_dir'] = {'group': 1, 'vals': FREESTREAM_WIND_DIRS}
max_downstream_dist = max(fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
min_downstream_dist = min(fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
# exclude most downstream turbine
upstream_turbine_indices = [t for t in range(n_turbines) if fi.floris.farm.turbine_map.coords[t].x1 < max_downstream_dist]
n_upstream_turbines = len(upstream_turbine_indices)
downstream_turbine_indices = [t for t in range(n_turbines) if fi.floris.farm.turbine_map.coords[t].x1 > min_downstream_dist]
n_downstream_turbines = len(downstream_turbine_indices)

for t_idx, t in enumerate(upstream_turbine_indices):
    case_inputs[f'yaw_angles_{t}'] = {'group': 2 + t_idx, 
                                      'vals': YAW_ANGLES}
    # step change in axial induction factor
    case_inputs[f'ax_ind_factors_{t}'] = {'group': 2 + n_upstream_turbines + t_idx, 
                                          'vals': AX_IND_FACTORS} 
                                                #    step_change([0.33, 0.22, 0.11], TOTAL_TIME, DT)]} #[0.22, 0.33, 0.67]}

case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix='.', namebase='wake_field', save_matrix=True)
case_list = np.repeat(case_list, N_SEEDS)
case_name_list = np.concatenate([[name] * N_SEEDS for name in case_name_list])
n_cases = len(case_name_list)

# **************************************** Simulation **************************************** #

def sim_func(case_idx, case):
    print(f'Simulating case #{case_idx}')
# for case_idx, case in enumerate(case_list):
    
    # Initialize
    fi = wfct.floris_interface.FlorisInterface(floris_dir)
    
    # def['ine wind speed time series
    ws_ts = {
        'TI': np.vstack([WS_TI] * int(TOTAL_TIME / DT)), # 5
        'mean': np.array(case['mean_wind_speed']).astype(float)
        # 'mean': np.concatenate([[mean_ws] * int((TOTAL_TIME / DT) / n_ws_steps) for mean_ws in np.linspace(8, 12, n_ws_steps)])
    }

    ws_ts['dev'] = (ws_ts['TI'] / 100) * ws_ts['mean']
    
    # define wind direction time series
    wd_ts = {
        'TI': np.vstack([WD_TI] * int(TOTAL_TIME / DT)), # 5
        'mean': np.array(case['mean_wind_dir']).astype(float)
    }
    
    wd_ts['dev'] = (wd_ts['TI'] / 100) * wd_ts['mean']
    
    # define yaw angle time series
    yaw_angles = DEFAULT_YAW_ANGLE * np.ones((int(TOTAL_TIME / DT), n_turbines))
    ai_factors = DEFAULT_AX_IND_FACTOR * np.ones((int(TOTAL_TIME / DT), n_turbines))
    yaw_angles[:, upstream_turbine_indices] = np.hstack([case[f'yaw_angles_{t}'] for t in upstream_turbine_indices]) # np.tile([case[f'yaw_angles_{t}'] for t in upstream_turbine_indices], (int(TOTAL_TIME / DT), 1))
    ai_factors[:, upstream_turbine_indices] = np.hstack([case[f'ax_ind_factors_{t}'] for t in upstream_turbine_indices]) # np.tile([case[f'ax_ind_factors_{t}'] for t in upstream_turbine_indices], (int(TOTAL_TIME / DT), 1))
    
    fi.reinitialize_flow_field(wind_speed=ws_ts['mean'][0], wind_direction=wd_ts['mean'][0]) 
    fi.calculate_wake(yaw_angles=yaw_angles[0, :], axial_induction=ai_factors[0, :])
    
    # lists that will be needed for visualizationsd
    freestream_wind_speed = []
    freestream_wind_dir = []
    turbine_wind_speeds = [[] for t in range(n_turbines)]
    turbine_wind_dirs = [[] for t in range(n_downstream_turbines)]
    turbine_turb_intensities = [[] for t in range(n_downstream_turbines)]
    time = np.arange(0, TOTAL_TIME, DT)
    horizontal_planes = []
    y_planes = []
    cross_planes = []
    
    for tt, sim_time in enumerate(time):
        if sim_time % 100 == 0:
            print("Simulation Time:", sim_time, "For Case:", case_idx)
        
        # fi.floris.farm.flow_field.mean_wind_speed = ws_ts['mean'][tt, 0]

        ws = np.random.normal(loc=ws_ts['mean'][tt], scale=ws_ts['dev'][tt])[0] #np.random.uniform(low=8, high=8.3)
        wd = np.random.normal(loc=wd_ts['mean'][tt], scale=wd_ts['dev'][tt])[0]

        fi.floris.farm.flow_field.mean_wind_speed = ws

        freestream_wind_speed.append(ws)
        freestream_wind_dir.append(wd)
        
        fi.reinitialize_flow_field(wind_speed=ws, wind_direction=wd, sim_time=sim_time)
        
        # calculate dynamic wake computationally
        fi.calculate_wake(sim_time=sim_time, yaw_angles=yaw_angles[tt], axial_induction=ai_factors[tt])
        
        if case_idx == 0:
            horizontal_planes.append(fi.get_hor_plane(x_resolution=200, y_resolution=100, height=90.0)) # horizontal plane at hub-height
            y_planes.append(fi.get_y_plane(x_resolution=200, z_resolution=100, y_loc=0.0)) # vertical plane parallel to freestream wind direction
            cross_planes.append(fi.get_cross_plane(y_resolution=100, z_resolution=100, x_loc=630.0)) # vertical plane parallel to turbine disc plane  
        
        for t_idx, t in enumerate(downstream_turbine_indices):
            # turbine_wind_speeds[t_idx].append(fi.floris.farm.wind_map.turbine_wind_speed[t])
            turbine_wind_speeds[t].append(fi.floris.farm.turbines[t].average_velocity)
            # turbine_wind_dirs[t_idx].append(fi.floris.farm.wind_map.turbine_wind_direction[t])
            # turbine_wind_dirs[t_idx].append(fi.floris.farm.wind_direction[t]) # TODO how to collect this accurately ?
            # turbine_turb_intensities[t_idx].append(fi.floris.farm.turbulence_intensity[t])

        for t_idx, t in enumerate(upstream_turbine_indices):
            turbine_wind_speeds[t].append(fi.floris.farm.turbines[t].average_velocity)
        
        # for t, turbine in enumerate(fi.floris.farm.turbines):
        #     turbine_powers[t].append(turbine.power/1e6)

        # NOTE: at this point, can also use measure other quantities like average velocity at a turbine, etc.
        # powers.append(sum([turbine.power for turbine in fi.floris.farm.turbines])/1e6)
        
        # calculate steady-state wake computationally
        # fi.calculate_wake(yaw_angles=yaw_angles[tt], axial_induction=ai_factors[tt])

        # NOTE: at this point, can also use measure other quantities like average velocity at a turbine, etc.
        # true_powers.append(sum([turbine.power for turbine in fi.floris.farm.turbines])/1e6)

    turbine_wind_speeds = np.array(turbine_wind_speeds).T
    # turbine_wind_dirs = np.array(turbine_wind_dirs).T
    # turbine_wind_dirs[turbine_wind_dirs > 180] = turbine_wind_dirs[turbine_wind_dirs > 180] - 360
    yaw_angles = np.array(yaw_angles)
    ai_factors = np.array(ai_factors)
    turbine_turb_intensities = np.array(turbine_turb_intensities).T
        
    # save case data as dataframe
    wake_field_data = {
        'Time': time,
        'FreestreamWindSpeed': freestream_wind_speed,
        'FreestreamWindDir': freestream_wind_dir
    }
    for t_idx, t in enumerate(upstream_turbine_indices + downstream_turbine_indices):
        wake_field_data = {**wake_field_data,
            f'TurbineWindSpeeds_{t}': turbine_wind_speeds[:, t],
            f'YawAngles_{t}': yaw_angles[:, t_idx],
            f'AxIndFactors_{t}': ai_factors[:, t_idx]             
        }

    wake_field_df = pd.DataFrame(data=wake_field_data)
    # wake_field_df.attrs['mean_wind_speed'] = case['mean_wind_speed']
    # wake_field_df.attrs['mean_wind_dir'] = case['mean_wind_dir']
    
    # for t in range(n_turbines):
    #     wake_field_df.attrs[f'yaw_angles_{t}'] = case[f'yaw_angles_{t}']
    #     wake_field_df.attrs[f'ai_factors_{t}'] = case[f'ax_ind_factors_{t}']
    
    # export case data to csv
    wake_field_df.to_csv(os.path.join(save_dir, f'case_{case_idx}.csv'))
    
    return wake_field_df, horizontal_planes, y_planes, cross_planes

def plot_ts(dfs, horizontal_planes, y_planes, cross_planes, upstream_turbine_indices, downstream_turbine_indices):
    # Plot vs. time
    n_plots = 4
    fig_ts, ax_ts = plt.subplots(n_plots, 1, sharex=True) #len(case_list), 5)
    ax_ts = ax_ts.flatten()
    
    n_cases = len(dfs)
    for case_idx in range(len(dfs)):
        if case_idx == 0 or n_cases < 5:
            
            time = dfs[case_idx]['Time']
            freestream_wind_speed = dfs[case_idx]['FreestreamWindSpeed']
            freestream_wind_dir = dfs[case_idx]['FreestreamWindDir']
            us_turbine_wind_speeds = np.hstack([dfs[case_idx][f'TurbineWindSpeeds_{t}'][:, np.newaxis] for t in upstream_turbine_indices])
            yaw_angles = np.hstack([dfs[case_idx][f'YawAngles_{t}'][:, np.newaxis] for t in upstream_turbine_indices])
            ai_factors = np.hstack([dfs[case_idx][f'AxIndFactors_{t}'][:, np.newaxis] for t in upstream_turbine_indices])
            ds_turbine_wind_speeds = np.hstack([dfs[case_idx][f'TurbineWindSpeeds_{t}'][:, np.newaxis] for t in downstream_turbine_indices])
            
            ax_ts[0].plot(time, freestream_wind_speed, label=f'Freestream Case {case_idx}')
            ax_ts[1].plot(time, freestream_wind_dir, label=f'Freestream Case {case_idx}')
            
            for t_idx, t in enumerate(upstream_turbine_indices):
                ax_ts[0].plot(time, us_turbine_wind_speeds[:, t_idx], label=f'US Turbine {t} Case {case_idx}')
                ax_ts[2].plot(time, yaw_angles[:, t_idx], label=f'US Turbine {t} Case {case_idx}')
                ax_ts[3].plot(time, ai_factors[:, t_idx], label=f'US Turbine {t} Case {case_idx}')
            
            for t_idx, t in enumerate(downstream_turbine_indices):
                ax_ts[0].plot(time, ds_turbine_wind_speeds[:, t_idx], label=f'DS Turbine {t} Case {case_idx}')
                # ax_ts[1].plot(time, turbine_wind_dirs[:, t_idx], label=f'DS Turbine {t}')
                # ax_ts[4].plot(time, turbine_turb_intensities[:, t_idx], label=f'DS Turbine {t}')
            
            ax_ts[0].set(title='Wind Speed [m/s]')
            ax_ts[0].legend()
            ax_ts[1].set(title='Wind Dir [deg]')
            ax_ts[1].legend()
            ax_ts[2].set(title='Yaw Angles [deg]')
            ax_ts[2].legend()
            ax_ts[3].set(title='Ax Ind Factors [-]')
            ax_ts[3].legend()
            # ax_ts[4].set(title='Turbulence Intensities [-]')
            # ax_ts[4].legend()
            for ax in ax_ts:
                ax.set(xticks=time[0:-1:int(60 // DT)], xlabel='Time [s]')
        
            # Animated Plot of Cut Planes
            # for hp, yp, cp in zip(horizontal_planes, y_planes, cross_planes):
            #     if len(hp) == 0:
            #         continue
            #     fig_anim, ax_anim = plt.subplots(3, 1)
            #     ax_anim[0].set(title='Horizontal')
            #     ax_anim[1].set(title='Streamwise Plane')
            #     ax_anim[2].set(title='Cross-stream Plane')
            #     def cut_plane_chart(i: int):
            #         ax_anim[0].clear()
            #         ax_anim[1].clear()
            #         ax_anim[2].clear()
            #         visualize_cut_plane(hp[i], ax=ax_anim[0])
            #         visualize_cut_plane(yp[i], ax=ax_anim[1])
            #         visualize_cut_plane(cp[i], ax=ax_anim[2])
                
                # animator_cut_plane = ani.FuncAnimation(fig_anim, cut_plane_chart, frames=int(TOTAL_TIME // DT))
                # # plt.show()
                # vid_writer = ani.FFMpegWriter(fps=100)
                # animator_cut_plane.save(os.path.join(fig_dir, 'cut_plane_vid.mp4'), writer=vid_writer)
    
if __name__ == '__main__':
    # if not DEBUG:
    pool = Pool()    
    res = pool.starmap(sim_func, zip(range(len(case_list)), case_list))
    dfs, horizontal_planes, y_planes, cross_planes = [r[0] for r in res], [r[1] for r in res], [r[2] for r in res], [r[3] for r in res]
    pool.close()
                                                                                                         
    # plot_ts(dfs, horizontal_planes, y_planes, cross_planes, upstream_turbine_indices, downstream_turbine_indices)
    
    plt.show()
    plt.savefig(os.path.join(fig_dir, 'wake_data.png'))