# TODO:
#  1) Set up deterministic MPC for 2 turbine system
#  2) Set up deterministic MPC with GP mean (trained offline) for 2 turbine system
#  3) Set up deterministic MPC with GP mean (trained offline + online) for 2 turbine system
#  4) Set up stochastic MPC with GP probability dist (trained offline + online) for 2 turbine system
#  5) Set up stochastic MPC with GP probability dist (trained offline + online) for 6 turbine system

# Formulate A, Bu, Bd matrices for wind farm (Lucy paper)

from sklearn.metrics import jaccard_score
import floridyn_special as fi
from scipy.optimize import minimize
import numpy as np
import matplotlib.pyplot as plt


class MPC:
    def __init__(self, system, 
                 stage_cost_func, terminal_cost_func,
                 stage_cost_jac_func, terminal_cost_jac_func,
                 stage_cost_hess_func, terminal_cost_hess_func,   
                 stage_ineq_constraint_func, term_ineq_constraint_func,
                 stage_ineq_constraint_jac_func, term_ineq_constraint_jac_func, 
                 horizon_length, dt_control):
        self.system = system
        self.stage_cost_func = stage_cost_func
        self.terminal_cost_func = terminal_cost_func
        self.stage_cost_jac_func = stage_cost_jac_func
        self.terminal_cost_jac_func = terminal_cost_jac_func
        self.stage_cost_hess_func = stage_cost_hess_func
        self.terminal_cost_hess_func = terminal_cost_hess_func
        self.stage_cost_func = stage_cost_func
        self.terminal_cost_func = terminal_cost_func
        self.stage_ineq_constraint_func = stage_ineq_constraint_func
        self.term_ineq_constraint_func = term_ineq_constraint_func
        self.stage_ineq_constraint_jac_func = stage_ineq_constraint_jac_func
        self.term_ineq_constraint_jac_func = term_ineq_constraint_jac_func
        
        
        self.horizon_length = horizon_length
        self.dt_control = dt_control
        if self.dt_control < self.system.dt:
            raise ValueError('dt_control must be greater or equal to system.dt')

        self.n_states = system.n_states
        self.n_ctrl_inputs = system.n_ctrl_inputs
        self.n_disturbances = system.n_disturbances
        self.n_outputs = system.n_outputs
        self.n_opt_vars = (self.n_states + self.n_ctrl_inputs) * horizon_length
        
        self.n_stage_constraints = len(self.stage_ineq_constraint_func(np.zeros(self.n_states), np.zeros(self.n_ctrl_inputs)))
        self.n_term_constraints = len(self.term_ineq_constraint_func(np.zeros(self.n_states)))
        
        self.x_pred_idx = slice(0, self.n_states * self.horizon_length)
        self.u_pred_idx = slice(self.n_states * self.horizon_length, (self.n_states * self.horizon_length) + (self.n_ctrl_inputs * self.horizon_length))
        u_opt_idx = np.arange(self.n_opt_vars)[self.u_pred_idx][:self.n_ctrl_inputs]
        self.u_opt_idx = slice(u_opt_idx[0], u_opt_idx[-1] + 1)
        x_shifted_idx = np.arange(self.n_opt_vars)[self.x_pred_idx][self.n_states:]
        self.x_shifted_idx = slice(x_shifted_idx[0], x_shifted_idx[-1] + 1)
        u_shifted_idx = np.arange(self.n_opt_vars)[self.u_pred_idx][self.n_ctrl_inputs:]
        self.u_shifted_idx = slice(u_shifted_idx[0], u_shifted_idx[-1] + 1)
        
        self.x0 = np.zeros(self.n_states)
        
        self.opt_var_labels = []
        for j in range(self.horizon_length):
            self.opt_var_labels = self.opt_var_labels + [f'x_{i}({j})' for i in range(self.n_states)]
        for j in range(self.horizon_length):
            self.opt_var_labels = self.opt_var_labels + [f'u_{i}({j})' for i in range(self.n_ctrl_inputs)]
        
    def initialize(self):
        pass
        # self.x_traj = [x0]
        # self.u_traj = []
        # self.k = 0
    
    def terminate(self):
        pass
    
    def get_x_stage(self, opt_vars, j):
        if j > self.horizon_length:
            raise Exception('stage index cannot be greater than horizon length')
        
        if j == 0:
            return self.x0
        else:
            return opt_vars[self.x_pred_idx][(j - 1) * self.n_states:j * self.n_states]
    
    def get_u_stage(self, opt_vars, j):
        if j > self.horizon_length - 1:
            raise Exception('stage index cannot be greater than horizon length - 1')
        
        return opt_vars[self.u_pred_idx][j * self.n_ctrl_inputs:(j + 1) * self.n_ctrl_inputs]
        
    def cost_func(self, opt_vars):
        return sum(self.stage_cost_func(self.get_x_stage(opt_vars, j), self.get_u_stage(opt_vars, j)) for j in range(self.horizon_length)) \
            + self.terminal_cost_func(self.get_x_stage(opt_vars, self.horizon_length))
    
    def cost_jac_func(self, opt_vars):
        
        # get jacobian for each stage and split between drvt wrt states and control inputs
        cost_jac_vals = np.zeros(self.n_opt_vars)
        for j in range(self.horizon_length):
            stage_cost_jac_vals = self.stage_cost_jac_func(self.get_x_stage(opt_vars, j), self.get_u_stage(opt_vars, j))
            state_terms = stage_cost_jac_vals[:self.n_states]
            ctrl_input_terms = stage_cost_jac_vals[self.n_states:]
            if j > 0:
                # only include x1, x2, ..., xN terms
                cost_jac_vals[self.x_pred_idx][j * self.n_states:(j + 1) * self.n_states] = state_terms
                
            cost_jac_vals[self.u_pred_idx][j * self.n_ctrl_inputs:(j + 1) * self.n_ctrl_inputs] = ctrl_input_terms
        cost_jac_vals[self.x_pred_idx][(self.horizon_length - 1) * self.n_states:self.horizon_length * self.n_states] \
            = self.terminal_cost_jac_func(self.get_x_stage(opt_vars, self.horizon_length))
        return cost_jac_vals
    
    def cost_hess_func(self, opt_vars):
        # get hessian for each stage and split between drvt wrt states and control inputs
        cost_hess_vals = np.zeros((self.n_opt_vars, self.n_opt_vars))
        
        for j in range(self.horizon_length):
            stage_cost_hess_vals = self.stage_cost_hess_func(self.get_x_stage(opt_vars, j), self.get_u_stage(opt_vars, j))
            state_terms = stage_cost_hess_vals[:self.n_states, :self.n_states]
            ctrl_input_terms = stage_cost_hess_vals[self.n_states:, self.n_states:]
            if j > 0:
                # only include x1, x2, ..., xN terms
                cost_hess_vals[self.x_pred_idx, self.x_pred_idx][j * self.n_states:(j + 1) * self.n_states, j * self.n_states:(j + 1) * self.n_states] \
                    = state_terms
                
            cost_hess_vals[self.u_pred_idx, self.u_pred_idx][j * self.n_ctrl_inputs:(j + 1) * self.n_ctrl_inputs, j * self.n_ctrl_inputs:(j + 1) * self.n_ctrl_inputs] \
                = ctrl_input_terms
            
        cost_hess_vals[self.x_pred_idx, self.x_pred_idx][(self.horizon_length - 1) * self.n_states:self.horizon_length * self.n_states, (self.horizon_length - 1) * self.n_states:self.horizon_length * self.n_states] \
            = self.terminal_cost_jac_func(self.get_x_stage(opt_vars, self.horizon_length))
            
        return cost_hess_vals
    
    def dyn_state_constraint_func(self, opt_vars, d_pred):
        dyn_state_con_vals = []
        for j in range(self.horizon_length):
            dyn_state_con_vals = np.concatenate([dyn_state_con_vals,
              (self.get_x_stage(opt_vars, j + 1) - self.system.dyn_state_eqn(self.get_x_stage(opt_vars, j), self.get_u_stage(opt_vars, j), d_pred[j, :])[0])])
        return np.array(dyn_state_con_vals)
    
    # def dyn_state_constraint_jac_func(self, opt_vars):
    #     dyn_state_constraint_jac_vals = np.zeros(((self.n_states * self.horizon_length), self.n_opt_vars))
        
    #     for j in range(self.horizon_length):
    #         dyn_state_constraint_jac_vals = -self.system.dyn_state_eqn_jac(self.get_x_stage(opt_vars, j), self.get_u_stage(opt_vars, j))
    #         state_terms = dyn_state_constraint_jac_vals[:, :self.n_states]
    #         ctrl_input_terms = dyn_state_constraint_jac_vals[:, self.n_states:]
    #         if j > 0:
    #             # only include x1, x2, ..., xN terms
    #             dyn_state_constraint_jac_vals[:, self.x_pred_idx][j * self.n_states:(j + 1) * self.n_states, 
    #                                                          j * self.n_states:(j + 1) * self.n_states] = state_terms
                
    #         dyn_state_constraint_jac_vals[:, self.u_pred_idx][j * self.n_states:(j + 1) * self.n_states, 
    #                                                      j * self.n_ctrl_inputs:(j + 1) * self.n_ctrl_inputs] = ctrl_input_terms
            
    #         dyn_state_constraint_jac_vals[:, self.x_pred_idx][j * self.n_states:(j + 1) * self.n_states,
    #                                                           (j + 1) * self.n_states:(j + 2) * self.n_states] = np.ones(self.n_states)
    #     return np.array(dyn_state_constraint_jac_vals)
        
    def ineq_constraint_func(self, opt_vars):
        ineq_con_vals = np.zeros((self.n_stage_constraints * self.horizon_length) + self.n_term_constraints)
        for j in range(self.horizon_length):
            ineq_con_vals[j * self.n_stage_constraints:(j + 1) * self.n_stage_constraints] \
                = self.stage_ineq_constraint_func(self.get_x_stage(opt_vars, j), self.get_u_stage(opt_vars, j))
        if self.n_term_constraints > 0:
            ineq_con_vals[self.horizon_length * self.n_stage_constraints:] = self.term_ineq_constraint_func(self.get_x_stage(opt_vars, self.horizon_length))
        return ineq_con_vals
    
    def ineq_constraint_jac_func(self, opt_vars):
            
        ineq_constraint_jac_vals = np.zeros(((self.n_stage_constraints * self.horizon_length) 
                                             + self.n_term_constraints, self.n_opt_vars))
        
        if self.n_stage_constraints + self.n_term_constraints == 0:
            return ineq_constraint_jac_vals
        
        for j in range(self.horizon_length):
            stage_ineq_constraint_jac_vals = self.stage_ineq_constraint_jac_func(self.get_x_stage(opt_vars, j), self.get_u_stage(opt_vars, j))
            state_terms = stage_ineq_constraint_jac_vals[:, :self.n_states]
            ctrl_input_terms = stage_ineq_constraint_jac_vals[:, self.n_states:]
            if j > 0:
                # only include x1, x2, ..., xN terms
                ineq_constraint_jac_vals[:, self.x_pred_idx][j * self.n_stage_constraints:(j + 1) * self.n_stage_constraints, 
                                                             j * self.n_states:(j + 1) * self.n_states] = state_terms
                
            ineq_constraint_jac_vals[:, self.u_pred_idx][j * self.n_stage_constraints:(j + 1) * self.n_stage_constraints, 
                                                         j * self.n_ctrl_inputs:(j + 1) * self.n_ctrl_inputs] = ctrl_input_terms
            
        ineq_constraint_jac_vals[:, self.x_pred_idx][self.horizon_length * self.n_stage_constraints:, 
                                                     (self.horizon_length - 1) * self.n_states:self.horizon_length * self.n_states] \
            = self.term_ineq_constraint_jac_func(self.get_x_stage(opt_vars, self.horizon_length))
        
        return ineq_constraint_jac_vals
    
    def run(self, last_opt_vars, d_pred):
        # opt_vars = [x1; x2; ...; xN; u0; u1; ...; uN-1]
        
        # COBYLA, SLSQP, trust-constr
        opt_vars_init = np.concatenate([last_opt_vars[self.x_shifted_idx], 
                                        np.zeros(self.n_states), 
                                        last_opt_vars[self.u_shifted_idx],
                                        np.zeros(self.n_ctrl_inputs)])
        
        # print(self.dyn_state_constraint_func(opt_vars_init, d_pred))
        # print(self.ineq_constraint_func(last_opt_vars))
        # print(self.ineq_constraint_jac_func(last_opt_vars))
        
        res = minimize(self.cost_func, x0=opt_vars_init, #method='SLSQP',
                       jac=self.cost_jac_func, hess=self.cost_hess_func,
                       constraints=[
                           {'type': 'eq', 
                            'fun': lambda x: self.dyn_state_constraint_func(x, d_pred), 
                            # 'jac': self.dyn_state_constraint_jac_func
                            },
                           {'type': 'ineq', 
                            'fun': lambda x: -self.ineq_constraint_func(x), 
                            'jac': lambda x: -self.ineq_constraint_jac_func(x)}
                           ]
                       )
        
        # get solution
        opt_vars = res.x
        u_opt = opt_vars[self.u_opt_idx]
        cost = res.fun
        cost_jac = res.jac
        ineq_constraint = self.ineq_constraint_func(opt_vars)
        
        # assert (np.abs(cost - self.cost_func(opt_vars)) < 1e-7).all()
        # assert (np.abs(cost_jac - self.cost_jac_func(opt_vars)) < 1e-6).all()
        # print(res.success, u_opt, cost)
        
        return u_opt, opt_vars, cost, cost_jac, ineq_constraint
        
    def simulate(self, x0, d_traj, t_max, plot=True):
        
        x_traj = [x0]
        u_traj = []
        y_traj = []
        cost_traj = []
        cost_jac_traj = []
        d_traj = np.vstack(d_traj)
        ineq_constraint_traj = []
        
        time_idx = np.arange(int(t_max // self.system.dt))
        opt_vars = np.zeros(self.n_opt_vars)
        
        for k in time_idx:
            self.x0 = x_traj[-1]
            d_pred = d_traj[k:k + self.horizon_length, :]
            
            # if it is the controller sampling time, run the mpc
            if (k * self.system.dt) % self.dt_control == 0:
                u_opt, opt_vars, cost, cost_jac, ineq_constraint = self.run(opt_vars, d_pred)
            
            # plug u_opt into system to get new state
            x1, y0 = self.system.dyn_state_eqn(self.x0, u_opt, d_pred[0, :])
            self.x0 = x1
            x_traj.append(x1)
            u_traj.append(u_opt)
            y_traj.append(y0)
            cost_traj.append(cost)
            cost_jac_traj.append(cost_jac)
            ineq_constraint_traj.append(ineq_constraint)
        
        u_traj = np.vstack(u_traj)
        x_traj = np.vstack(x_traj)
        y_traj = np.vstack(y_traj)
        cost_traj = np.vstack(cost_traj)
        cost_jac_traj = np.vstack(cost_jac_traj)
        ineq_constraint_traj = np.vstack(ineq_constraint_traj)

        if plot:
            
            # plot states, control inputs, disturbances and outputs
            _, traj_axs = plt.subplots(sum([self.n_states > 0, self.n_ctrl_inputs > 0, self.n_disturbances > 0, self.n_outputs > 0]), 1, sharex=True)
            ax_idx = 0
            if self.n_states > 0:
                for i in range(self.n_states):
                    traj_axs[ax_idx].plot(time_idx * self.system.dt, x_traj[:-1, i], label=f'x_{i}')
                    traj_axs[ax_idx].set(ylabel=f'States')
                traj_axs[ax_idx].legend()
                ax_idx += 1
            
            if self.n_ctrl_inputs > 0:
                for i in range(self.n_ctrl_inputs):
                    traj_axs[ax_idx].plot(time_idx * self.system.dt, u_traj[:, i], label=f'u_{i}')
                    traj_axs[ax_idx].set(ylabel=f'Control Inputs')
                traj_axs[ax_idx].legend()
                ax_idx += 1
            
            if self.n_disturbances > 0:    
                for i in range(self.n_disturbances):
                    traj_axs[ax_idx].plot(time_idx * self.system.dt, d_traj[:, i], label=f'd_{i}')
                    traj_axs[ax_idx].set(ylabel=f'Disturbances')
                traj_axs[ax_idx].legend()
                ax_idx += 1
            
            if self.n_outputs > 0:
                for i in range(self.n_outputs):
                    traj_axs[ax_idx].plot(time_idx * self.system.dt, y_traj[:, i], label=f'y_{i}')
                    traj_axs[ax_idx].set(ylabel=f'Outputs')
                traj_axs[ax_idx].legend()
                
            traj_axs[-1].set(xlabel='Time [s]')
            plt.show()
            
            # plot cost
            _, cost_axs = plt.subplots(2, 1, sharex=True)
            cost_axs[0].plot(time_idx * self.system.dt, cost_traj)
            cost_axs[0].set(ylabel=f'$J$')
            for i in range(self.n_opt_vars):
                cost_axs[1].plot(time_idx * self.system.dt, cost_jac_traj[:, i], label=self.opt_var_labels[i])
            cost_axs[1].set(ylabel=r'$\frac{dJ}{dz}$')
            cost_axs[1].legend()
            plt.show()
            
            # plot constraints
            _, constraint_axs = plt.subplots(self.n_stage_constraints + self.n_term_constraints, 1)
            for i in range(self.n_stage_constraints + self.n_term_constraints):
                constraint_axs[i].plot(time_idx, ineq_constraint_traj[:, i], label=f'g_{i}')
            
            plt.show()     
             
        return x_traj, u_traj, y_traj, cost_traj, ineq_constraint_traj
    

class StochasticMPC:
    def __init__(self, system, 
                 stage_cost_func, terminal_cost_func,
                 stage_cost_jac_func, terminal_cost_jac_func,
                 stage_cost_hess_func, terminal_cost_hess_func,   
                 stage_ineq_constraint_func, term_ineq_constraint_func,
                 stage_ineq_constraint_jac_func, term_ineq_constraint_jac_func, 
                 horizon_length, dt_control):
        
        
        super().__init__(system, 
                        stage_cost_func, terminal_cost_func,
                        stage_cost_jac_func, terminal_cost_jac_func,
                        stage_cost_hess_func, terminal_cost_hess_func,   
                        stage_ineq_constraint_func, term_ineq_constraint_func,
                        stage_ineq_constraint_jac_func, term_ineq_constraint_jac_func, 
                        horizon_length, dt_control)