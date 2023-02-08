from system import System
from mpc import MPC
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import norm
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
from postprocessing import plot_training_data, plot_score, plot_std_evolution, plot_ts, plot_error_ts, plot_k_train_evolution
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
	SIM_SAVE_DIR = f'/scratch/summit/aohe7145/wake_gp/{FARM_LAYOUT}_wake_field_simulations'
	TS_SAVE_DIR = f'/scratch/summit/aohe7145/wake_gp/{FARM_LAYOUT}_wake_field_ts_data'
	FLORIS_DIR = f'./{FARM_LAYOUT}_floris_input.json'
	BASE_MODEL_FLORIS_DIR = f'./{FARM_LAYOUT}_base_model_floris_input.json'
	DATA_DIR = f'/scratch/summit/aohe7145/wake_gp/data'
	FIG_DIR = f'/scratch/summit/aohe7145/wake_gp/figs'
	SCALARS_DIR = '/scratch/summit/aohe7145/wake_gp//scalars'

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
                     'lines.markersize': 10})

RUN_MPC = False
TRAIN_ONLINE = True
TRAIN_OFFLINE = False
FIT_ONLINE = True

parser = argparse.ArgumentParser()
parser.add_argument('-d', '--debug', action='store_true', default=False)
parser.add_argument('-p', '--parallel', action='store_true', default=False)
parser.add_argument('-rs', '--run_simulations', action='store_true', default=False)
parser.add_argument('-gp', '--generate_plots', action='store_true', default=False)
parser.add_argument('case_ids', type=int, nargs='+')
args = parser.parse_args()
DEBUG = args.debug
PARALLEL = args.parallel
RUN_SIMULATIONS = args.run_simulations
GENERATE_PLOTS = args.generate_plots

TMAX = 300 if DEBUG else 1200
N_TOTAL_DATASETS = 5 if DEBUG else 500

if not os.path.exists(os.path.join(SIM_SAVE_DIR)):
	os.makedirs(SIM_SAVE_DIR)

if not os.path.exists(os.path.join(FIG_DIR)):
	os.makedirs(FIG_DIR)

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
		print(stage_cost_func(x0, u_traj[0, :]))  # expect
		print(term_cost_func(x0))
		print(stage_cost_jac_func(x0, u_traj[0, :]))
		print(stage_cost_hess_func(x0, u_traj[0, :]))
		print(term_cost_jac_func(x0))
		print(term_cost_hess_func(x0))
		
		# Define Constraints
		# TODO test adding one at a time
		stage_ineq_constraint_func = lambda xj, uj: []  # [xj[0] - 10, -xj[0] + 10] #, uj[0] - 20, -uj[0] + 20]
		term_ineq_constraint_func = lambda xN: []  # [xN[0] - 10, -xN[0] + 10]
		stage_ineq_constraint_jac_func = lambda xj, uj: np.zeros(
			(0, sys.n_states + sys.n_ctrl_inputs))  # np.vstack([[1, 0, 0], [-1, 0, 0]])#, [0, 0, 1], [0, 0, -1]])
		term_ineq_constraint_jac_func = lambda xN: np.zeros((0, sys.n_states))  # np.vstack([[1, 0], [-1, 0]])
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
