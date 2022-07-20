import matplotlib.pyplot as plt
import numpy as np

class System:
    def __init__(self, n_states, n_ctrl_inputs, n_disturbances, n_outputs, diff_eqn_sol_func, dt=None, linear_matrices=None, nonlinear_func=None, is_discrete=True):
        self.n_states = n_states
        self.n_ctrl_inputs = n_ctrl_inputs
        self.n_disturbances = n_disturbances
        self.n_outputs = n_outputs
        self.diff_eqn_sol_func = getattr(DiffEqnSol(), diff_eqn_sol_func)
        
        if linear_matrices is not None:
            self.sys_type = 'linear'
            self.linear_matrices = linear_matrices
            for key, val in self.linear_matrices.items():
                if val is None:
                    if key == 'A':
                        self.linear_matrices['A'] = np.zeros((self.n_states, self.n_states))
                    elif key == 'Bu':
                        self.linear_matrices['Bu'] = np.zeros((self.n_states, self.n_ctrl_inputs))
                    elif key == 'Bd':
                        self.linear_matrices['Bd'] = np.zeros((self.n_states, self.n_disturbances))
                    elif key == 'C':
                        self.linear_matrices['C'] = np.zeros((self.n_outputs, self.n_states))
                    elif key == 'Du':
                        self.linear_matrices['Du'] = np.zeros((self.n_outputs, self.n_ctrl_inputs))
                    elif key == 'Dd':
                        self.linear_matrices['Dd'] = np.zeros((self.n_outputs, self.n_disturbances))
                else:
                    self.linear_matrices[key] = np.array(val)
            self.dt = dt
        elif nonlinear_func is not None:
            self.sys_type = 'nonlinear'
            self.nonlinear_func = nonlinear_func
        else:
            raise Exception('Either linear_matrices or nonlinear_func must have a value that is not None')
    
        self.is_discrete = is_discrete
        
    def dyn_state_eqn(self, x0, u0, d0):
        
        if self.sys_type == 'linear':
            y0 = self.linear_matrices['C'] @ x0 + self.linear_matrices['Du'] @ u0 + self.linear_matrices['Dd'] @ d0
            if self.is_discrete:
                x1 = self.linear_matrices['A'] @ x0 + self.linear_matrices['Bu'] @ u0 + self.linear_matrices['Bd'] @ d0
            else:
                dxdt_func = lambda x0, u0, d0: self.linear_matrices['A'] @ x0 + self.linear_matrices['Bu'] @ u0 + self.linear_matrices['Bd'] @ d0
                x1 = self.diff_eqn_sol_func(x0, u0, d0, dxdt_func, self.dt)
        elif self.sys_type == 'nonlinear':
                pass
        
        return x1, y0
    
    # def dyn_state_eqn_jac(self, x0, u0, d0):
    #     if self.sys_type == 'linear':
    #         if self.is_discrete:
    #             pass
    #         else:
    #             if self.diff.eqn_sol_func.__name__ == 'forward_diff':
                    
    #     elif self.sys_type == 'nonlinear':
    #         pass

    def simulate(self, x0, u_traj, d_traj, plot=True):
        x_traj = [x0]
        y_traj = []
        k_max = len(u_traj)
        # assert t_max == len(d_traj)
        time_idx = np.arange(k_max)
        
        for k in time_idx:
            x1, y0 = self.dyn_state_eqn(x_traj[-1], u_traj[k], d_traj[k])
            x_traj.append(x1)
            y_traj.append(y0)
        
        u_traj = np.vstack(u_traj)
        d_traj = np.vstack(d_traj)
        x_traj = np.vstack(x_traj)
        y_traj = np.vstack(y_traj)

        if plot:
            fig, axs = plt.subplots(self.n_ctrl_inputs + self.n_disturbances + self.n_states + self.n_outputs, 1, sharex=True)
            for i in range(len(axs)):
                if i < self.n_ctrl_inputs:
                    axs[i].plot(time_idx * self.dt, u_traj[:, i])
                    axs[i].set(ylabel=f'u({i})')
                elif i < self.n_ctrl_inputs + self.n_disturbances:
                    axs[i].plot(time_idx * self.dt, d_traj[:, i - self.n_ctrl_inputs])
                    axs[i].set(ylabel=f'd({i - self.n_ctrl_inputs})')
                elif i < self.n_ctrl_inputs + self.n_disturbances + self.n_states:
                    axs[i].plot(time_idx * self.dt, x_traj[:-1, i - self.n_ctrl_inputs - self.n_disturbances])
                    axs[i].set(ylabel=f'x({i - self.n_ctrl_inputs - self.n_disturbances})')
                elif i < self.n_ctrl_inputs + self.n_disturbances + self.n_states + self.n_outputs:
                    axs[i].plot(time_idx * self.dt, y_traj[:, i - self.n_ctrl_inputs - self.n_disturbances - self.n_states])
                    axs[i].set(ylabel=f'y({i - self.n_ctrl_inputs - self.n_disturbances - self.n_states})')
            axs[-1].set(xlabel='Time [s]')
            plt.show()
        
        return x_traj, y_traj       

class DiffEqnSol:
    def __init__(self):
        pass
    
    def forward_difference(self, x0, u0, d0, dxdt_func, dt):
        return x0 + dt * dxdt_func(x0, u0, d0)

    def RK4(self, x0, u0, d0, dxdt_func, dt):
        k1 = dxdt_func(x0, u0, d0)
        k2 = dxdt_func(x0 + (dt * (k1 / 2)), u0, d0)
        k3 = dxdt_func(x0 + (dt * (k2 / 2)), u0, d0)
        k4 = dxdt_func(x0 + (dt * k3), u0, d0)
        return x0 + (1 / 6) * (k1 + (2 * k2) + (2 * k3) + k4) * dt