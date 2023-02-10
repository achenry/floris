import matplotlib.pyplot as plt
import numpy as np
from _collections import defaultdict

def plot_prediction_vs_input(ax, gpr_fit, inputs, input_labels, X_norm, y_norm, X_scalar, y_scalar, learning_turbine_index, dataset_type):
            
    input_indices = [input_labels.index(f) for f in inputs]
    
    # for d, dataset_type in enumerate(['train', 'test']):
    mean, std = gpr_fit.predict(X_norm, return_std=True)
    mean = y_scalar.inverse_transform(mean[:, np.newaxis]).squeeze()
    std = std / y_scalar.scale_
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
        axs[-1, col_idx].set(xlabel='Time [s]', xticks=time[start_indices[0]::300])
    
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


def plot_error_ts(system_fi, simulation_results, sim_indices):
    """
   Squarred error (true turbine effective wind speed vs. GP estimate)
   for given simulations for each downstream turbines vs. time
    Returns:

    """
    turbine_sim_error = defaultdict(list)
    # {'train': np.zeros(len(simulation_results['train']), , len(system_fi.downstream_turbine_indices)),
    #                      'test': None}
    
    for dataset_type in ['train', 'test']:
        for sim_idx in sim_indices:
            sim = simulation_results[dataset_type][sim_idx]
            # for i, ds_idx in enumerate(system_fi.downstream_turbine_indices):
            turbine_sim_error[dataset_type].append(np.square(np.subtract(sim['true'], sim['pred'])))
    
    n_plots = 2 if len(turbine_sim_error['test']) else 1
    error_fig, error_ax = plt.subplots(n_plots, 1)
    
    for ax_idx, dataset_type in enumerate(['train', 'test']):
        sim = turbine_sim_error[dataset_type]
        if not len(sim):
            continue
            
        for sim_idx in range(len(sim_indices)):
            time_steps = list(range(len(sim[sim_idx])))
            for t, t_idx in enumerate(system_fi.downstream_turbine_indices):
                error_ax[ax_idx].scatter(time_steps, sim[sim_idx][:, t], label=f'Turbine {t_idx}')

    error_ax[-1].set(xlabel='Time Step')
    error_ax[-1].legend()
    return error_fig


def plot_score(system_fi, turbine_score, turbine_score_mean, turbine_score_std, score_type):
    """
   RMSE mean and std (true turbine effective wind speed vs. GP estimate) over all simulations for
    each downstream turbine
    Returns:
    
    """
    score_fig, score_ax = plt.subplots(1, 1)
    
    dataset_type = 'train'
    # score_ax.errorbar(x=system_fi.downstream_turbine_indices,
    #                           y=turbine_score_mean[dataset_type],
    #                           yerr=turbine_score_std[dataset_type],
    #                           fmt='o', color='orange',
    #                           ecolor='lightgreen', elinewidth=5, capsize=10)
    
    c1 = 'orange'
    c2 = 'green'
    lw = 3
    score_ax.boxplot(x=np.array(turbine_score[dataset_type]).T, patch_artist=True,
                     boxprops=dict(facecolor='white', color=c1, linewidth=lw), # facecolor
                     capprops=dict(color=c2, linewidth=lw),
                     whiskerprops=dict(color=c1, linewidth=lw),
                     flierprops=dict(color=c1, markeredgecolor=c2),
                     medianprops=dict(color=c2, linewidth=lw)
                     )
    # score_fig.show()
    
    # score_ax[ax_idx].set(
    #     title=f'Downstream Turbine Effective Wind Speed {score_type.upper()} Score over all {dataset_type.capitalize()}ing Simulations [m/s]'
    # )

    score_ax.set(xlabel='Turbine Number')
    return score_fig


def plot_ts(all_ds_indices, ds_indices, simulation_results, sim_indices, time):
    """
   GP estimate, true value, noisy measurements of
    effective wind speeds of downstream turbines vs.
    time for one dataset
    Returns:

    """
    
    ts_fig, ts_ax = plt.subplots(len(ds_indices),
                                 len(sim_indices['train']),
                                 sharex=True, sharey=True)
    
    dataset_type = 'train'
    min_val = np.infty
    max_val = -np.infty
    for row_idx, ds in enumerate(ds_indices):
        ax_idx = -1
        if not len(sim_indices[dataset_type]):
            continue
        
        for j, sim_idx in enumerate(sim_indices[dataset_type]):
            ax_idx += 1
            ts_ax[-1, ax_idx].set(xlabel='Time [s]', xticks=list(range(0, time[-1] + 300, 300)))
            ts_ax[ax_idx, 0].set(ylabel='[m/s]')
            
            sim_data = simulation_results[dataset_type][sim_idx]
            
            # if sim_idx == 1 and row_idx == 1:
            #     print("stop!")
            
            ds_idx = all_ds_indices.index(ds)
            # ts_ax[ds_idx, ax_idx].scatter(time, simulation_results[dataset_type][sim_idx]['true'][:, ds_idx],
            #                    color='orange', label=f'True', marker="o")
            ts_ax[row_idx, ax_idx].plot(time, sim_data['pred'][:, ds_idx],
                               color='green', label=f'Predicted Mean')
            # ts_ax[ds_idx, ax_idx].plot(time, simulation_results[dataset_type][sim_idx]['modeled'][:, ds_idx],
            #                    color='purple', label=f'Base Modeled')
            ts_ax[row_idx, ax_idx].scatter(time, sim_data['meas'][:, ds_idx],
                                          color='red',
                                  label=f'Measurements', marker="^")
            ts_ax[row_idx, ax_idx].fill_between(time,
                                               sim_data['pred'][:, ds_idx] - sim_data['std'][:, ds_idx],
                                       sim_data['pred'][:, ds_idx] + sim_data['std'][:, ds_idx],
                                       alpha=0.1, label=f'Predicted Std. Dev.', color='lightgreen')

            min_val = min(min_val, np.nanmin([ln.get_ydata() for ln in ts_ax[row_idx, ax_idx].get_lines()]))
            max_val = max(max_val, np.nanmax([ln.get_ydata() for ln in ts_ax[row_idx, ax_idx].get_lines()]))

    for row_idx, ds in enumerate(ds_indices):
        ax_idx = -1
        ds_idx = all_ds_indices.index(ds)
        for j, sim_idx in enumerate(sim_indices[dataset_type]):
            ax_idx += 1
            training_start_idx = np.where(~np.isnan(sim_data['pred'][:, ds_idx]))[0][0]
            ts_ax[row_idx, ax_idx].plot([training_start_idx, training_start_idx], [min_val, max_val], linestyle='--',
                                       color='#1f77b4')

            # ts_ax[ax_idx].set(
            #     title=f'Downstream Turbine Effective Wind Speeds for {dataset_type.capitalize()}ing Simulation {j} [m/s]')

    ts_ax[-1, -1].legend(loc='upper right')
    return ts_fig


def plot_std_evolution(all_ds_indices, ds_indices, simulation_results, sim_indices, time):
    """
    plot evolution of sum of predicted variance at grid test points for middle column downstream turbines vs online training time
    Returns:

    """
    # for each simulation, each time gp.add_training_data is called,
    # the predicted variance is computed for a grid of test points
    std_fig, std_ax = plt.subplots(len(sim_indices['train']), len(ds_indices), sharex=True, sharey=True)

    dataset_type = 'train'
    min_val = np.infty
    max_val = -np.infty
    ax_idx = -1
    for j, sim_idx in enumerate(sim_indices[dataset_type]):
        ax_idx += 1
        for col_idx, ds in enumerate(ds_indices):
            ds_idx = all_ds_indices.index(ds)
            std_ax[ax_idx, col_idx].plot(time, simulation_results[dataset_type][sim_idx]['test_std'][:, ds_idx],
                                        color='green')

            # std_ax[ax_idx].set(
            #     title=f'Downstream Turbine Effective Wind Speed Standard Deviation '
            #           f'vs. Time for {dataset_type.capitalize()}ing Simulation {j} [m/s]')
            
            std_ax[0, col_idx].set(title=f'Turbine {ds}', ylabel='[m/s]')
            std_ax[-1, col_idx].set(xlabel='Time [s]')
            min_val = min(min_val, np.nanmin([ln.get_ydata() for ln in std_ax[ax_idx, col_idx].get_lines()]))
            max_val = max(max_val, np.nanmax([ln.get_ydata() for ln in std_ax[ax_idx, col_idx].get_lines()]))
            std_ax[ax_idx, col_idx].set(xticks=np.arange(time[0], time[-1] + 300, 300))

    ax_idx = -1
    for j, sim_idx in enumerate(sim_indices[dataset_type]):
        ax_idx += 1
        for col_idx, ds in enumerate(ds_indices):
            ds_idx = all_ds_indices.index(ds)
            training_start_idx = np.where(~np.isnan(simulation_results[dataset_type][sim_idx]['test_std'][:, ds_idx]))[0][0]
            std_ax[ax_idx, col_idx].plot([training_start_idx, training_start_idx], [min_val, max_val], linestyle='--',
                                        color='#1f77b4')
            
    return std_fig


def plot_k_train_evolution(all_ds_indices, ds_indices, simulation_results, sim_indices, time):
    """
    plot evolution of sum of predicted variance at grid test points for middle column downstream turbines vs online training time
    Returns:

    """
    # for each simulation, each time gp.add_training_data is called,
    # the predicted variance is computed for a grid of test points
    k_train_fig, k_train_ax = plt.subplots(len(sim_indices['train']), 1,
                                   sharex=True, sharey=True)
    
    ax_idx = -1
    dataset_type = 'train'
    
    for j, sim_idx in enumerate(sim_indices[dataset_type]):
        for ds in ds_indices:
            ax_idx += 1
            ds_idx = all_ds_indices.index(ds)
            [k_train[ds_idx] if len(k_train) else [] for k_train in simulation_results[dataset_type][sim_idx]['k_train']]
            
            k_train_ax[ax_idx].scatter(time, simulation_results[dataset_type][sim_idx]['k_train'][:, ds_idx],
                                label=f'Turbine {ds}')
            
            # k_train_ax[ax_idx].set(
            #     title=f'Downstream TurbineTime-Step Content of Training Data '
            #           f'vs. Time for {dataset_type.capitalize()}ing Simulation {j} [m/s]')
            k_train_ax[ax_idx].set(title=f'Simulation {j}', ylabel='[m/s]')
    
    # k_train_ax[0].legend(loc='center left')
    k_train_ax[-1].set(xlabel='Time [s]')
    return k_train_fig

def compute_score(system_fi, simulation_results, score_type):
    turbine_score_mean = defaultdict(list)
    turbine_score_std = defaultdict(list)
    turbine_sim_score = defaultdict(list)
    sim_score = {}

    for dataset_type in ['train', 'test']:
    
        if len(simulation_results[dataset_type]) == 0:
            continue
        
        # for each downstream turbine
        for i, ds_idx in enumerate(system_fi.downstream_turbine_indices):
            
            # compute the rmse of the turbine effective wind speed error for each simulation
            turbine_scores = []
            for sim in simulation_results[dataset_type]:
                y_true = sim['true'][:, i]
                y_pred = sim['pred'][:, i]
                if score_type == 'rmse':
                    score = np.nanmean((y_true - y_pred)**2)**0.5
                elif score_type == 'r2':
                    score = 1 - (np.nansum((y_true - y_pred)**2) / np.nansum((y_true - np.nanmean(y_true))**2))
            
                if np.isnan(score):
                    print('oh no')
                    
                turbine_scores.append(score)
                
            turbine_sim_score[dataset_type].append(turbine_scores)
            turbine_score_mean[dataset_type].append(np.mean(turbine_sim_score[dataset_type][i]))
            turbine_score_std[dataset_type].append(np.std(turbine_sim_score[dataset_type][i]))

        # for each simulation, compute mean rmse over all downstream turbines
        sim_score[dataset_type] = np.mean(turbine_sim_score[dataset_type], axis=0)
        assert sim_score[dataset_type].shape[0] == len(simulation_results[dataset_type])

    return sim_score, turbine_sim_score, turbine_score_mean, turbine_score_std