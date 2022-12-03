import matplotlib.pyplot as plt
import numpy as np

def plot_prediction_vs_input(ax, gpr_fit, inputs, input_labels, X_norm, y_norm, X_scalar, y_scalar, learning_turbine_index, dataset_type):
            
    input_indices = [input_labels.index(f) for f in inputs]
    
    # for d, dataset_type in enumerate(['train', 'test']):
    mean, std = gpr_fit.predict(X_norm, return_std=True)
    mean = y_scalar.inverse_transform(mean[:, np.newaxis]).squeeze()
    std = y_scalar.inverse_transform(std[:, np.newaxis]).squeeze()
    X = X_scalar.inverse_transform(X_norm)
    y = y_scalar.inverse_transform(y_norm)
    
    for ax_idx, input_idx in enumerate(input_indices):
        sort_idx = np.argsort(X[:, input_idx])

        ax[ax_idx].plot(X[sort_idx, input_idx], mean[sort_idx], label='Predicted')
        ax[ax_idx].fill_between(X[sort_idx, input_idx], 
                        mean[sort_idx] - std[sort_idx], 
                        mean[sort_idx] + std[sort_idx], 
                        alpha=0.1, label='Std. Dev')
        
        ax[ax_idx].scatter(X[sort_idx, input_idx], y[sort_idx], 
                            linestyle='dashed', color='black', label='Measurements')
        
        ax[ax_idx].set(title=f'TurbineWindSpeeds_{learning_turbine_index} - {dataset_type}', 
                        xlabel=f'{input_labels[input_idx]}', 
                        xlim=[min(X[:, input_idx]), max(X[:, input_idx])])
        
        # ax[ax_idx, d].legend()
    plt.subplots_adjust(wspace=0.6, hspace=0.4)

def plot_training_data(measurements_df, animated=False):

    inputs = [inp for inp in measurements_df.columns
              if any(inp_type in inp
                     for inp_type in ['TurbineWindSpeeds', 'YawAngles', 'AxIndFactors'])]
    fig, ax = plt.subplots(1, len(inputs), figsize=(49, 49))
    # fig, ax = plt.subplots(1, 1, figsize=(35, 49))
    ax = ax.flatten()
    for i, inp in enumerate(inputs):
        ax[i].scatter(np.ones(len(measurements_df[inp].index)), measurements_df[inp])
        ax[i].set(title=inp)

    fig.show()

def plot_prediction_vs_time(ax, time, X_ts, y_ts):
    ax_idx = 0
    for input_idx, (input_label, input_ts) in enumerate(X_ts.items()):
        ax[ax_idx].set(title=input_label)
        if input_ts.ndim == 1:
            ax[ax_idx].plot(time, input_ts)
        else:
            for i in range(input_ts.shape[1]):
                ax[ax_idx].plot(time, input_ts[:, i])
        ax_idx += 1
    
    n_time_steps = len(output_ts['y_true']) # due to effective_k_delay
    for output_idx, (output_label, output_ts) in enumerate(y_ts.items()):
        ax[ax_idx].set(title=output_label, xlabel='Time [s]')
        ax[ax_idx].plot(time[-n_time_steps:], output_ts['y_true'], label='True')
        ax[ax_idx].plot(time[-n_time_steps:], output_ts['y_pred'], label='Predicted Mean')
        ax[ax_idx].fill_between(time[-n_time_steps:], output_ts['y_pred'] - output_ts['y_std'], output_ts['y_pred'] + output_ts['y_std'], alpha=0.1, label='Predicted Std. Dev.')
        ax[ax_idx].legend(loc='center left')

def plot_measurements_vs_time(axs, time, X, y, input_labels, input_types, delay_indices, output_labels):
    n_datapoints = X.shape[0]
    start_indices = sorted([0, n_datapoints] + list(np.where(np.diff(time) < 0)[0] + 1))
    
    for ts_idx in range(len(start_indices) - 1):
        start_t_idx = start_indices[ts_idx]
        end_t_idx = start_indices[ts_idx + 1]
        
        for row_idx, delay_idx in enumerate(delay_indices):
            for col_idx, input_type in enumerate(input_types):
                input = f'{input_type}_minus{delay_idx}'
                input_idx = input_labels.index(input)

                # ax = fig.add_subplot(gs[row_idx, col_idx])
                ax = axs[row_idx, col_idx]
                ax.plot(time[start_t_idx:end_t_idx], X[start_t_idx:end_t_idx, input_idx], label=f'Case {ts_idx}')
                ax.set(title=input)
                
        for output_idx, output_label in enumerate(output_labels):
            # ax = fig.add_subplot(gs[row_idx, output_idx])
            row_idx = len(delay_indices) + int(output_idx // len(input_types))
            ax = axs[row_idx, output_idx]
            ax.plot(time[start_t_idx:end_t_idx], y[start_t_idx:end_t_idx, output_idx], label=f'Case {ts_idx} Output {output_label}')
            ax.set(title=output_label)
        
    axs[0, 0].legend()
    for col_idx in range(len(input_types)):
        axs[-1, col_idx].set(xlabel='Time [s]', xticks=time[start_indices[0]::100])
    
    plt.subplots_adjust(wspace=0.6, hspace=0.4)

def plot_raw_measurements_vs_time(ax, plotting_dfs, input_types):
    for df in plotting_dfs:
        for input in [col for col in df.columns if col != 'Time']:
            for input_type_idx, input_type in enumerate(input_types):
                if input_type in input:
                    row_idx = input_type_idx
                
                    ax[row_idx].plot(df['Time'], df[input])
                    ax[row_idx].set(title=input)
            
            ax[-1].set(xlabel='Time [s]')
    plt.subplots_adjust(wspace=0.6, hspace=0.4)

def plot_measurements(full_offline_measurements_df):
    # plot noisy vs. noise-free measurements of Turbine Wind Speed
    fig, ax = plt.subplots(1, len(system_fi.floris.farm.turbines))
    for t_idx in range(len(system_fi.floris.farm.turbines)):
        ax[t_idx].scatter(full_offline_measurements_df['Time'],
                          full_offline_measurements_df[f'TurbineWindSpeeds_{t_idx}'],
                          color='red', label='True')
        ax[t_idx].scatter(noisy_measurements_df['Time'],
                          noisy_measurements_df[f'TurbineWindSpeeds_{t_idx}'],
                          color='blue', label='Noisy')
        ax[t_idx].set(title=f'Turbine {t_idx} Wind Speed [m/s] measurements', xlabel='Time [s]')
    ax[0].legend()
    plt.show()


def plot_score(system_fi, turbine_score_mean, turbine_score_std, score_type):
    """
   RMSE mean and std (true turbine effective wind speed vs. GP estimate) over all simulations for
    each downstream turbine
    Returns:
    
    """
    n_plots = 2 if len(turbine_score_mean['test']) else 1
    score_fig, score_ax = plt.subplots(n_plots, 1)

    for ax_idx, dataset_type in enumerate(['train', 'test']):
        if not len(turbine_score_mean[dataset_type]):
            continue
            
        score_ax[ax_idx].errorbar(x=system_fi.downstream_turbine_indices,
                                  y=turbine_score_mean[dataset_type],
                                  yerr=turbine_score_std[dataset_type],
                                  fmt='o', color='orange',
                                  ecolor='lightgreen', elinewidth=5, capsize=10)
        # score_ax[ax_idx].set(
        #     title=f'Downstream Turbine Effective Wind Speed {score_type.upper()} Score over all {dataset_type.capitalize()}ing Simulations [m/s]'
        # )

    score_ax[-1].set(xlabel='Turbine Index')
    return score_fig


def plot_ts(tmax, ds_indices, simulation_results, sim_indices):
    """
   GP estimate, true value, noisy measurements of
    effective wind speeds of downstream turbines vs.
    time for one dataset
    Returns:

    """
    time = np.arange(tmax)
    ts_fig, ts_ax = plt.subplots(len(ds_indices),
                                 len(sim_indices['train']) + len(sim_indices['test']),
                                 sharex=True, sharey=True)

    for ds_idx, ds in enumerate(ds_indices):
        ax_idx = -1
        for i, dataset_type in enumerate(['train', 'test']):
            if not len(sim_indices[dataset_type]):
                continue
            
            for j, sim_idx in enumerate(sim_indices[dataset_type]):
                ax_idx += 1
                ts_ax[-1, ax_idx].set(xlabel='Time [s]', xticks=list(range(0, tmax, 50)))
                
                ts_ax[ds_idx, ax_idx].plot(time, simulation_results[dataset_type][sim_idx]['true'][:, ds_idx],
                                   label=f'True')
                # ts_ax[ds_idx, ax_idx].plot(time, simulation_results[dataset_type][sim_idx]['pred'][:, ds_idx],
                #                    label=f'Predicted Mean')
                # ts_ax[ds_idx, ax_idx].plot(time, simulation_results[dataset_type][sim_idx]['modeled'][:, ds_idx],
                #                    label=f'Base Modeled')
                # ts_ax[ds_idx, ax_idx].scatter(time, simulation_results[dataset_type][sim_idx]['meas'][:, ds_idx], c='r',
                #                       label=f'Measurements')
                # ts_ax[ds_idx, ax_idx].fill_between(time, simulation_results[dataset_type][sim_idx]['pred'][:, ds_idx]
                #                            - simulation_results[dataset_type][sim_idx]['std'][:, ds_idx],
                #                            simulation_results[dataset_type][sim_idx]['pred'][:, ds_idx]
                #                            + simulation_results[dataset_type][sim_idx]['std'][:, ds_idx],
                #                            alpha=0.1, label=f'Predicted Std. Dev.')

                # ts_ax[ax_idx].set(
                #     title=f'Downstream Turbine Effective Wind Speeds for {dataset_type.capitalize()}ing Simulation {j} [m/s]')

    ts_ax[0, 0].legend(loc='center left')
    ts_fig.show()
    return ts_fig


def plot_std_evolution(tmax, ds_indices, simulation_results, sim_indices):
    """
    plot evolution of sum of predicted variance at grid test points for middle column downstream turbines vs online training time
    Returns:

    """
    # for each simulation, each time gp.add_training_data is called,
    # the predicted variance is computed for a grid of test points
    time = np.arange(tmax)
    std_fig, std_ax = plt.subplots(len(sim_indices['train']) + len(sim_indices['test']), 1,
                                   sharex=True, sharey=True)

    ax_idx = -1
    for i, dataset_type in enumerate(['train', 'test']):
        if not len(sim_indices[dataset_type]):
            continue
        
        for j, sim_idx in enumerate(sim_indices[dataset_type]):
            ax_idx += 1
            for ds_idx, ds in enumerate(ds_indices):
                std_ax[ax_idx].plot(time, simulation_results[dataset_type][sim_idx]['test_std'][:, ds_idx],
                                    label=f'Turbine {ds}')

                std_ax[ax_idx].set(
                    title=f'Downstream Turbine Effective Wind Speed Standard Deviation '
                          f'vs. Time for {dataset_type.capitalize()}ing Simulation {j} [m/s]')

    std_ax[0].legend(loc='center left')
    std_ax[-1].set(xlabel='Time [s]')
    return std_fig