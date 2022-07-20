from system import System
from mpc import MPC
import numpy as np
from scipy.linalg import block_diag
from scipy.stats import norm

if __name__ == '__main__':
    
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
    # print(stage_cost_func(x0, u_traj[0, :]))
    # print(term_cost_func(x0))
    # print(stage_cost_jac_func(x0, u_traj[0, :]))
    # print(stage_cost_hess_func(x0, u_traj[0, :]))
    # print(term_cost_jac_func(x0))
    # print(term_cost_hess_func(x0))
    
    # Define Constraints
    # TODO test adding one at a time
    stage_ineq_constraint_func = lambda xj, uj: [xj[0] - 10, -xj[0] + 10] #, uj[0] - 20, -uj[0] + 20]
    term_ineq_constraint_func = lambda xN: [] # [xN[0] - 10, -xN[0] + 10]
    stage_ineq_constraint_jac_func = lambda xj, uj: np.vstack([[1, 0, 0], [-1, 0, 0]])#, [0, 0, 1], [0, 0, -1]])
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
    
    # opt_vars = np.zeros(mpc.n_opt_vars)
    # print(mpc.cost_func(opt_vars))
    # print(mpc.cost_jac_func(opt_vars))
    # print(mpc.cost_hess_func(opt_vars))
    # print(mpc.ineq_constraint_func(opt_vars))
    # print(mpc.ineq_constraint_jac_func(opt_vars), np.array(mpc.ineq_constraint_jac_func(opt_vars)).shape)
    
    mpc.simulate(x0, d_traj, t_max=(len(d_traj) - horizon_length) * sys.dt)
    
    pass