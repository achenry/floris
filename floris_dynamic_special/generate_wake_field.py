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
#import ffmpeg

# **************************************** Parameters **************************************** #

# total simulation time
total_time = 600 # ten minutes
dt = 1.0 # DOESN'T WORK WELL WITH OTHER DT VALUES
DEFAULT_AX_IND_FACTOR = 0.67
DEFAULT_YAW_ANGLE = 0

# Test varying wind speed and direction? Or test one at a time? 
# Step change and random variation will be added.
test_varying_ws = True
test_varying_wd = True

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
elif sys.platform == 'linux':
    save_dir = '/scratch/ahenry/2turb_wake_field_cases'
    
if not os.path.exists(save_dir):
    os.mkdir(save_dir)


# **************************************** GENERATE TIME-VARYING FREESTREAM WIND SPEED/DIRECTION, YAW ANGLE, TURBINE TOPOLOGY SWEEP **************************************** #
# TODO - alt generate DLCs using turbsim

case_inputs = {}
case_inputs['mean_wind_speed'] = {'group': 0, 'vals': np.linspace(8, 12, 3)}
case_inputs['mean_wind_dir'] = {'group': 1, 'vals': np.linspace(250, 290, 3)}
max_downstream_dist = max(fi.floris.farm.turbine_map.coords[t].x1 for t in range(n_turbines))
upstream_turbine_indices = []
for t in range(n_turbines):
    # if not most downstream turbine
    if fi.floris.farm.turbine_map.coords[t].x1 < max_downstream_dist:
        upstream_turbine_indices.append(t)
        case_inputs[f'yaw_angles_{t}'] = {'group': 2 + t, 'vals': [0]} #np.linspace(0, 15, 3)}
        case_inputs[f'ax_ind_factors_{t}'] = {'group': 2 + n_turbines + t, 'vals': [0, 0.33, 0.67]}

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
    
    # define wind speed time series
    ws_ts = {
        'TI': np.array([0] * int(total_time / dt)), # 5
        'mean': np.array([case['mean_wind_speed']] * int(total_time / dt))
        # 'mean': np.concatenate([[mean_ws] * int((total_time / dt) / n_ws_steps) for mean_ws in np.linspace(8, 12, n_ws_steps)])
    }

    ws_ts['dev'] = (ws_ts['TI'] / 100) * ws_ts['mean']
    
    # define wind direction time series
    wd_ts = {
        'TI': np.array([0] * int(total_time / dt)), # 5
        'mean': np.array([case['mean_wind_dir']] * int(total_time / dt))
    }
    
    wd_ts['dev'] = (wd_ts['TI'] / 100) * wd_ts['mean']
    
    # define yaw angle time series
    yaw_angles = DEFAULT_YAW_ANGLE * np.ones((int(total_time / dt), n_turbines))
    ai_factors = DEFAULT_AX_IND_FACTOR * np.ones((int(total_time / dt), n_turbines))
    yaw_angles[:, upstream_turbine_indices] = np.tile([case[f'yaw_angles_{t}'] for t in upstream_turbine_indices], (int(total_time / dt), 1))
    ai_factors[:, upstream_turbine_indices] = np.tile([case[f'ax_ind_factors_{t}'] for t in upstream_turbine_indices], (int(total_time / dt), 1))
    
    # lists that will be needed for visualizationsd
    freestream_wind_speed = []
    freestream_wind_dir = []
    turbine_wind_speeds = [[] for t in range(n_turbines)]
    turbine_wind_dirs = [[] for t in range(n_turbines)]
    turbine_turb_intensities = [[] for t in range(n_turbines)]
    time = np.arange(0, total_time, dt)
    horizontal_planes = []
    y_planes = []
    cross_planes = []
    
    for tt, sim_time in enumerate(time):
        if sim_time % 100 == 0:
            print("Simulation Time:", sim_time, "For Case:", case_idx)
        
        fi.floris.farm.flow_field.mean_wind_speed = ws_ts['mean'][tt]
            
        ws = np.random.normal(loc=ws_ts['mean'][tt], scale=ws_ts['dev'][tt]) #np.random.uniform(low=8, high=8.3)
        wd = np.random.normal(loc=wd_ts['mean'][tt], scale=wd_ts['dev'][tt])
        
        freestream_wind_speed.append(ws)
        freestream_wind_dir.append(wd)
        
        fi.reinitialize_flow_field(wind_speed=ws, wind_direction=wd, sim_time=sim_time)
        
        # calculate dynamic wake computationally
        fi.calculate_wake(sim_time=sim_time, yaw_angles=yaw_angles[tt], axial_induction=ai_factors[tt])
        
        if case_idx == 0:
            horizontal_planes.append(fi.get_hor_plane(x_resolution=200, y_resolution=100, height=90.0)) # horizontal plane at hub-height
            y_planes.append(fi.get_y_plane(x_resolution=200, z_resolution=100, y_loc=0.0)) # vertical plane parallel to freestream wind direction
            cross_planes.append(fi.get_cross_plane(y_resolution=100, z_resolution=100, x_loc=630.0)) # vertical plane parallel to turbine disc plane  
        
        for t in range(n_turbines):
            turbine_wind_speeds[t].append(fi.floris.farm.wind_map.turbine_wind_speed[t])
            turbine_wind_dirs[t].append(fi.floris.farm.wind_map.turbine_wind_direction[t])
            turbine_turb_intensities[t].append(fi.floris.farm.wind_map.turbine_turbulence_intensity[t])
        
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
    turbine_wind_dirs[turbine_wind_dirs > 180] = turbine_wind_dirs[turbine_wind_dirs > 180] - 360
    yaw_angles = np.array(yaw_angles)
    ai_factors = np.array(ai_factors)
    turbine_turb_intensities = np.array(turbine_turb_intensities).T
    
    # Plot vs. time
    if case_idx == 0:
        fig_ts, ax_ts = plt.subplots(1, 5) #len(case_list), 5)
        ax_ts = ax_ts.flatten()

        ax_ts[(case_idx * 5) + 0].plot(time, freestream_wind_speed, label='Freestream')
        ax_ts[(case_idx * 5) + 1].plot(time, freestream_wind_dir, label='Freestream')
        for t in range(n_turbines):
            ax_ts[(case_idx * 5) + 0].plot(time, turbine_wind_speeds[:, t], label=f'Turbine {t}')
            ax_ts[(case_idx * 5) + 1].plot(time, turbine_wind_dirs[:, t], label=f'Turbine {t}')
            ax_ts[(case_idx * 5) + 2].plot(time, yaw_angles[:, t], label=f'Turbine {t}')
            ax_ts[(case_idx * 5) + 3].plot(time, ai_factors[:, t], label=f'Turbine {t}')
            ax_ts[(case_idx * 5) + 4].plot(time, turbine_turb_intensities[:, t], label=f'Turbine {t}')
        
        ax_ts[0].set(title='Wind Speed [m/s]')
        ax_ts[0].legend()
        ax_ts[1].set(title='Wind Dir [deg]')
        ax_ts[1].legend()
        ax_ts[2].set(title='Yaw Angles [deg]')
        ax_ts[2].legend()
        ax_ts[3].set(title='Ax Ind Factors [-]')
        ax_ts[3].legend()
        ax_ts[4].set(title='Turbulence Intensities [-]')
        ax_ts[4].legend()
        plt.show()
    
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
            
        animator_cut_plane = ani.FuncAnimation(fig_anim, cut_plane_chart, interval=100)
        plt.show()
        
    # save case data as dataframe
    wake_field_data = {
        'Time': time,
        'FreestreamWindSpeed': freestream_wind_speed,
        'FreestreamWindDir': freestream_wind_dir
    }
    for t in range(n_turbines):
        wake_field_data = {**wake_field_data, 
            f'TurbineWindSpeeds_{t}': turbine_wind_speeds[:, t],
            f'TurbineWindDirs_{t}': turbine_wind_dirs[:, t],
            f'TurbineTurbIntens_{t}': turbine_turb_intensities[:, t],
            f'YawAngles_{t}': yaw_angles[:, t],
            f'AxIndFactors_{t}': ai_factors[:, t]             
        }
        
        
    wake_field_df = pd.DataFrame(data=wake_field_data)
    wake_field_df.attrs['mean_wind_speed'] = case['mean_wind_speed']
    wake_field_df.attrs['mean_wind_dir'] = case['mean_wind_dir']
    
    for t in range(n_turbines):
        wake_field_df.attrs[f'yaw_angles_{t}'] = case[f'yaw_angles_{t}']
        wake_field_df.attrs[f'ai_factors_{t}'] = case[f'ax_ind_factors_{t}']
    
    # export case data to csv
    wake_field_df.to_csv(os.path.join(save_dir, f'case_{case_idx}.csv'))
    
if __name__ == '__main__':
    pool = Pool()
    pool.starmap(sim_func, zip(range(0, len(case_list)), case_list))
    pool.close()