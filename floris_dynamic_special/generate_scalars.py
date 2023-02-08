from DownstreamTurbineGPR import GP_CONSTANTS
import os
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, Matern
import sys

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

if DEBUG:
    
    # construct case hierarchy
    default_kernel = lambda: ConstantKernel(constant_value_bounds=(1e-12, 1e12)) * RBF(
        length_scale_bounds=(1e-12, 1e12))
    default_kernel_idx = 0
    default_max_training_size = 30
    default_batch_size = 2
    default_noise_std = 0.01
    default_k_delay = 2
    
    KERNELS = [default_kernel()]
    MAX_TRAINING_SIZE_VALS = [default_max_training_size, int(default_max_training_size / 2)]
    NOISE_STD_VALS = [default_noise_std]
    K_DELAY_VALS = [default_k_delay]
    BATCH_SIZE_VALS = [default_batch_size]
    
    TMAX = 1200
    GP_CONSTANTS['PROPORTION_TRAINING_DATA'] = 1  # 6 / 9
    N_TOTAL_DATASETS = 1
    
    COMPUTE_SCALARS = False

else:
    KERNELS = [lambda: ConstantKernel(constant_value_bounds=(1e-12, 1e12)) * RBF(length_scale_bounds=(1e-12, 1e12)),
               lambda: ConstantKernel(constant_value_bounds=(1e-12, 1e12)) * Matern(length_scale_bounds=(1e-12, 1e12))]
    MAX_TRAINING_SIZE_VALS = [50, 100, 200, 400]
    NOISE_STD_VALS = [0.0001, 0.001, 0.01, 0.1]
    K_DELAY_VALS = [2, 4, 6, 8]
    BATCH_SIZE_VALS = [1, 2, 3, 4]
    
    TMAX = 1200
    GP_CONSTANTS['PROPORTION_TRAINING_DATA'] = 1  # 4 / 5
    N_TOTAL_DATASETS = 5  # TODO test 500 on RC
    
    # construct case hierarchy
    default_kernel = lambda: ConstantKernel(constant_value_bounds=(1e-12, 1e12)) * RBF(
        length_scale_bounds=(1e-12, 1e12))
    default_kernel_idx = 0
    default_max_training_size = 100
    default_batch_size = 10
    default_noise_std = 0.01
    default_k_delay = 6
    
    COMPUTE_SCALARS = False  # TODO test True on RC

cases = [{'kernel': default_kernel(), 'max_training_size': default_max_training_size,
          'noise_std': default_noise_std, 'k_delay': default_k_delay, 'batch_size': x} for x in BATCH_SIZE_VALS] + \
        [{'kernel': default_kernel(), 'max_training_size': default_max_training_size, 'noise_std': default_noise_std,
          'k_delay': x, 'batch_size': default_batch_size} for x in K_DELAY_VALS if x != default_k_delay] + \
        [{'kernel': default_kernel(), 'max_training_size': default_max_training_size, 'noise_std': x,
          'k_delay': default_k_delay, 'batch_size': default_batch_size} for x in NOISE_STD_VALS if
         x != default_noise_std] + \
        [{'kernel': default_kernel(), 'max_training_size': x, 'noise_std': default_noise_std,
          'k_delay': default_k_delay, 'batch_size': default_batch_size} for x in MAX_TRAINING_SIZE_VALS if
         x != default_max_training_size] + \
        [{'kernel': x(), 'max_training_size': default_max_training_size, 'noise_std': default_noise_std,
          'k_delay': default_k_delay, 'batch_size': default_batch_size} for i, x in enumerate(KERNELS) if
         i != default_kernel_idx]

CASE_IDS = list(range(len(cases))) if len(sys.argv) < 3 else [int(i) for i in sys.argv[2].split(',')]
cases = [cases[c] if c in CASE_IDS else None for c in range(len(cases))]

if not os.path.exists(os.path.join(SCALARS_DIR)):
    os.makedirs(SCALARS_DIR)