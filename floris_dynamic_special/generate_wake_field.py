# %matplotlib inline
'''
Generate 'true' wake field data for use in GP learning procedure
Inputs: Yaw Angles, Freestream Wind Velocity, Freestream Wind Direction, Turbine Topology
Need csv containing 'true' wake characteristics at each turbine (variables) at each time-step (rows).
'''

from defusedxml import DTDForbidden
import matplotlib.pyplot as plt
import matplotlib.animation as ani
import numpy as np
from floridyn_special import tools as wfct # Incoming con
import pandas as pd
from weis.aeroelasticse.CaseGen_General import CaseGen_General
from floridyn_special.tools.visualization import visualize_cut_plane
import pickle
import os
import sys
import multiprocessing as mp
from multiprocessing import Pool

if sys.platform == 'darwin':
    save_dir = './wake_field_cases'
    data_dir = './data'
    fig_dir = './figs'
elif sys.platform == 'linux':
    save_dir = '/scratch/ahenry/wake_field_cases'
    data_dir = '/scratch/ahenry/data'
    fig_dir = '/scratch/ahenry/figs'

# **************************************** Parameters **************************************** #

# total simulation time
DEBUG = True
dt = 1.0 # DOESN'T WORK WELL WITH OTHER DT VALUES
DEFAULT_AX_IND_FACTOR = 2 / 3
DEFAULT_YAW_ANGLE = 0

def step_change(vals, T, dt):
    step_vals = []
    k_change = int(T // dt // len(vals))
    for val in vals:
        for k in range(k_change):
            step_vals.append(val)
    
    while len(step_vals) < T // dt:
        step_vals.append(vals[-1])
        
    return np.vstack(step_vals)

# **************************************** Initialization **************************************** #
# Initialize the FLORIS interface fi
# For basic usage, the florice interface provides a simplified interface to
# the underlying classes
# floris_dir = "./2turb_floris_input.json"

floris_dir = "./2turb_floris_input.json"

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
if sys.platform == 'darwin':
    save_dir = './2turb_wake_field_cases'
    DEBUG = True
elif sys.platform == 'linux':
    save_dir = '/scratch/ahenry/2turb_wake_field_cases'
    DEBUG = False
    
if not os.path.exists(save_dir):
    os.mkdir(save_dir)

total_time = 600 if DEBUG else 600 # ten minutes

# **************************************** GENERATE TIME-VARYING FREESTREAM WIND SPEED/DIRECTION, YAW ANGLE, TURBINE TOPOLOGY SWEEP **************************************** #
# TODO - alt generate DLCs using turbsim

case_inputs = {}
case_inputs['mean_wind_speed'] = {'group': 0, 
                                  'vals': [step_change([8], total_time, dt)] if DEBUG else [step_change([val], total_time, dt) for val in np.linspace(8, 12, 3)]}
case_inputs['mean_wind_dir'] = {'group': 1, 
                                'vals': [step_change([270], total_time, dt)] if DEBUG else [step_change([val], total_time, dt) for val in np.linspace(250, 270, 3)]}
max_downstream_dist = max(fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
min_downstream_dist = min(fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
# exclude most downstream turbine
upstream_turbine_indices = [t for t in range(n_turbines) if fi.floris.farm.turbine_map.coords[t].x1 < max_downstream_dist]
n_upstream_turbines = len(upstream_turbine_indices)
downstream_turbine_indices = [t for t in range(n_turbines) if fi.floris.farm.turbine_map.coords[t].x1 > min_downstream_dist]
n_downstream_turbines = len(downstream_turbine_indices)

for t_idx, t in enumerate(upstream_turbine_indices):
    case_inputs[f'yaw_angles_{t}'] = {'group': 2 + t_idx, 
                                      'vals': [step_change([0], total_time, dt)]} #np.linspace(0, 15, 3)}
    # step change in axial induction factor
    case_inputs[f'ax_ind_factors_{t}'] = {'group': 2 + n_upstream_turbines + t_idx, 
                                          'vals': [step_change([0.22, 0.28, 0.34], total_time, dt), 
                                                   step_change([0.33, 0.41, 0.49], total_time, dt), 
                                                   step_change([0.66, 0.58, 0.5], total_time, dt)]} #[0.22, 0.33, 0.67]}

case_list, case_name_list = CaseGen_General(case_inputs, dir_matrix='.', namebase='wake_field', save_matrix=True)

# TODO - consider cases where each turbine is offline, still produces some wake? Can we simulate the wake from an offline turbine in FLORIS? - can I set turbine.aiset to imitate an offline turbine?
# cases['turbine_top'] = {'group': 0, 'vals': np.linspace(-25, 25, 3)}
# fi.floris.farm.turbine_map.coords[t].x1

# TODO for each case, save freestream_wind_speed_ts, freestream_wind_dir_ts, turbine_wind_speed_ts, turbine_wind_dir_ts, yaw_angles_ts and any other wake parameters that are available
# **************************************** Simulation **************************************** #


def sim_func(case_idx, case):

# for case_idx, case in enumerate(case_list):
    
    # Initialize
    fi = wfct.floris_interface.FlorisInterface(floris_dir)
    
    # def['ine wind speed time series
    ws_ts = {
        'TI': np.vstack([0] * int(total_time / dt)), # 5
        'mean': np.array(case['mean_wind_speed']).astype(float)
        # 'mean': np.concatenate([[mean_ws] * int((total_time / dt) / n_ws_steps) for mean_ws in np.linspace(8, 12, n_ws_steps)])
    }

    ws_ts['dev'] = (ws_ts['TI'] / 100) * ws_ts['mean']
    
    # define wind direction time series
    wd_ts = {
        'TI': np.vstack([0] * int(total_time / dt)), # 5
        'mean': np.array(case['mean_wind_dir']).astype(float)
    }
    
    wd_ts['dev'] = (wd_ts['TI'] / 100) * wd_ts['mean']
    
    # define yaw angle time series
    yaw_angles = DEFAULT_YAW_ANGLE * np.ones((int(total_time / dt), n_turbines))
    ai_factors = DEFAULT_AX_IND_FACTOR * np.ones((int(total_time / dt), n_turbines))
    yaw_angles[:, upstream_turbine_indices] = [case[f'yaw_angles_{t}'] for t in upstream_turbine_indices] # np.tile([case[f'yaw_angles_{t}'] for t in upstream_turbine_indices], (int(total_time / dt), 1))
    ai_factors[:, upstream_turbine_indices] = [case[f'ax_ind_factors_{t}'] for t in upstream_turbine_indices] # np.tile([case[f'ax_ind_factors_{t}'] for t in upstream_turbine_indices], (int(total_time / dt), 1))
    
    fi.reinitialize_flow_field(wind_speed=ws_ts['mean'][0], wind_direction=wd_ts['mean'][0]) 
    fi.calculate_wake()
    
    # lists that will be needed for visualizationsd
    freestream_wind_speed = []
    freestream_wind_dir = []
    turbine_wind_speeds = [[] for t in range(n_downstream_turbines)]
    turbine_wind_dirs = [[] for t in range(n_downstream_turbines)]
    turbine_turb_intensities = [[] for t in range(n_downstream_turbines)]
    time = np.arange(0, total_time, dt)
    horizontal_planes = []
    y_planes = []
    cross_planes = []
    
    for tt, sim_time in enumerate(time):
        if sim_time % 100 == 0:
            print("Simulation Time:", sim_time, "For Case:", case_idx)
        
        fi.floris.farm.flow_field.mean_wind_speed = ws_ts['mean'][tt, 0]
            
        ws = np.random.normal(loc=ws_ts['mean'][tt], scale=ws_ts['dev'][tt])[0] #np.random.uniform(low=8, high=8.3)
        wd = np.random.normal(loc=wd_ts['mean'][tt], scale=wd_ts['dev'][tt])[0]
        
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
            turbine_wind_speeds[t_idx].append(fi.floris.farm.turbines[t].average_velocity)
            # turbine_wind_dirs[t_idx].append(fi.floris.farm.wind_map.turbine_wind_direction[t])
            # turbine_wind_dirs[t_idx].append(fi.floris.farm.wind_direction[t]) # TODO how to collect this accurately ?
            # turbine_turb_intensities[t_idx].append(fi.floris.farm.turbulence_intensity[t])
        
        # for t, turbine in enumerate(fi.floris.farm.turbines):
        #     turbine_powers[t].append(turbine.power/1e6)

        # NOTE: at this point, can also use measure other quantities like average velocity at a turbine, etc.
        # powers.append(sum([turbine.power for turbine in fi.floris.farm.turbines])/1e6)
        
        # calculate steady-state wake computationally
        fi.calculate_wake(yaw_angles=yaw_angles[tt], axial_induction=ai_factors[tt])

        # NOTE: at this point, can also use measure other quantities like average velocity at a turbine, etc.
        # true_powers.append(sum([turbine.power for turbine in fi.floris.farm.turbines])/1e6)

    turbine_wind_speeds = np.array(turbine_wind_speeds).T    
    turbine_wind_dirs = np.array(turbine_wind_dirs).T
    # turbine_wind_dirs[turbine_wind_dirs > 180] = turbine_wind_dirs[turbine_wind_dirs > 180] - 360
    yaw_angles = np.array(yaw_angles)
    ai_factors = np.array(ai_factors)
    turbine_turb_intensities = np.array(turbine_turb_intensities).T
    
    # Plot vs. time
    if case_idx == 0:
        n_plots = 4
        fig_ts, ax_ts = plt.subplots(1, n_plots) #len(case_list), 5)
        ax_ts = ax_ts.flatten()

        ax_ts[0].plot(time, freestream_wind_speed, label='Freestream')
        ax_ts[1].plot(time, freestream_wind_dir, label='Freestream')
        
        for t_idx, t in enumerate(upstream_turbine_indices):
            ax_ts[2].plot(time, yaw_angles[:, t_idx], label=f'US Turbine {t}')
            ax_ts[3].plot(time, ai_factors[:, t_idx], label=f'US Turbine {t}')
        
        for t_idx, t in enumerate(downstream_turbine_indices):
            ax_ts[0].plot(time, turbine_wind_speeds[:, t_idx], label=f'DS Turbine {t}')
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
            ax.set(xticks=time, xlabel='Time [s]')
        plt.show()
        plt.savefig(os.path.join(fig_dir, 'wake_data.png'))
    
        # Animated Plot of Cut Planes
        fig_anim, ax_anim = plt.subplots(3, 1)
        ax_anim[0].set(title='Horizontal')
        ax_anim[1].set(title='Streamwise Plane')
        ax_anim[2].set(title='Cross-stream Plane')
        def cut_plane_chart(i: int):
            ax_anim[0].clear()
            ax_anim[1].clear()
            ax_anim[2].clear()
            visualize_cut_plane(horizontal_planes[i], ax=ax_anim[0])
            visualize_cut_plane(y_planes[i], ax=ax_anim[1])
            visualize_cut_plane(cross_planes[i], ax=ax_anim[2])
            
        animator_cut_plane = ani.FuncAnimation(fig_anim, cut_plane_chart, frames=int(total_time // dt))
        # plt.show()
        vid_writer = ani.FFMpegWriter(fps=100)
        animator_cut_plane.save(os.path.join(fig_dir, 'cut_plane_vid.mp4'), writer=vid_writer)
        
    # save case data as dataframe
    wake_field_data = {
        'Time': time,
        'FreestreamWindSpeed': freestream_wind_speed,
        'FreestreamWindDir': freestream_wind_dir
    }
    for t_idx, t in enumerate(upstream_turbine_indices):
        wake_field_data = {**wake_field_data, 
            f'YawAngles_{t}': yaw_angles[:, t_idx],
            f'AxIndFactors_{t}': ai_factors[:, t_idx]             
        }
    
    for t_idx, t in enumerate(downstream_turbine_indices):
        wake_field_data = {**wake_field_data, 
            f'TurbineWindSpeeds_{t}': turbine_wind_speeds[:, t_idx],
            # f'TurbineWindDirs_{t}': turbine_wind_dirs[:, t_idx],
            # f'TurbineTurbIntens_{t}': turbine_turb_intensities[:, t_idx]
        }
    
    wake_field_df = pd.DataFrame(data=wake_field_data)
    # wake_field_df.attrs['mean_wind_speed'] = case['mean_wind_speed']
    # wake_field_df.attrs['mean_wind_dir'] = case['mean_wind_dir']
    
    # for t in range(n_turbines):
    #     wake_field_df.attrs[f'yaw_angles_{t}'] = case[f'yaw_angles_{t}']
    #     wake_field_df.attrs[f'ai_factors_{t}'] = case[f'ax_ind_factors_{t}']
    
    # export case data to csv
    wake_field_df.to_csv(os.path.join(save_dir, f'case_{case_idx}.csv'))
    
if __name__ == '__main__':
    # if not DEBUG:
    pool = Pool()    
    pool.starmap(sim_func, zip(range(len(case_list)), case_list))
    pool.close()
    # else:
    #     for c in range(len(case_list)):
    #         sim_func(c, case_list[c])
    
