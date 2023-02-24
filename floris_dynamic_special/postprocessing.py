import matplotlib.pyplot as plt
import numpy as np
from _collections import defaultdict
from DownstreamTurbineGPR import WIND_SPEED_RANGE
from mpl_toolkits.axes_grid1 import make_axes_locatable
from floridyn.tools.visualization import plot_turbines_with_fi, visualize_cut_plane
import pandas as pd
import os

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
        ax[t_idx].set(title=f'T{t_idx} Wind Speed [m/s] measurements', xlabel='Time [s]')
    ax[0].legend()
    plt.show()


def plot_error_ts(all_ds_indices, ds_indices, simulation_results, time):
    """
   Relative Error (true turbine effective wind speed vs. GP estimate)
   for given simulations for each downstream turbines vs. time
    Returns:

    """
    error_fig, error_ax = plt.subplots(len(simulation_results),len(ds_indices),
                                 sharex=True)

    for row_idx, (sim_idx, sim_data) in enumerate(simulation_results):
        # std_ax[row_idx, 0].set_title(f'Sim\n{sim_idx}')
        error_ax[row_idx, 0].set_ylabel(f'Sim {sim_idx + 1}\n[%]', rotation=0, ha='right', labelpad=15.0, y=0.8)
        for col_idx, ds in enumerate(ds_indices):
            error_ax[0, col_idx].set_title(f'T{ds}')
            error_ax[-1, col_idx].set(xlabel='Time [s]', xticks=list(range(0, time[-1] + 600, 600)))
            ds_idx = all_ds_indices.index(ds)
            
            y_true = sim_data['true'][:, ds_idx]
            y_pred = sim_data['pred'][:, ds_idx]
            y_modeled = sim_data['modeled'][:, ds_idx]
            score = ((y_pred + y_modeled - y_true) / y_true) * 100
            
            error_ax[row_idx, col_idx].plot(time, score, label=f'T{ds}')
    
    # error_ax[0].legend(loc='upper center')
    return error_fig

def plot_score(system_fi, scores_df, score_type='rmse'):
    """
   RMSE mean and std (true turbine effective wind speed vs. GP estimate) over all simulations for
    each downstream turbine
    Returns:
    
    """
    score_fig, score_ax = plt.subplots(1, 1)
    
    # score_ax.errorbar(x=system_fi.downstream_turbine_indices,
    #                           y=turbine_score_mean[dataset_type],
    #                           yerr=turbine_score_std[dataset_type],
    #                           fmt='o', color='orange',
    #                           ecolor='lightgreen', elinewidth=5, capsize=10)
    
    c1 = 'orange'
    c2 = 'green'
    c3 = '#1f77b4'
    lw = 3
    x = scores_df.sort_values(by='Turbine')[['Turbine', score_type]]
    x = [x.loc[x['Turbine'] == t][score_type].to_list() for t in system_fi.downstream_turbine_indices]
    score_ax.boxplot(x=x, patch_artist=True,
                     boxprops=dict(facecolor='white', color=c1, linewidth=lw), # facecolor
                     capprops=dict(color=c2, linewidth=lw),
                     whiskerprops=dict(color=c2, linewidth=lw),
                     flierprops=dict(color=c1, markeredgecolor=c2),
                     medianprops=dict(color=c3, linewidth=lw)
                     )
    # score_fig.show()
    
    # score_ax[ax_idx].set(
    #     title=f'Downstream Turbine Effective Wind Speed {score_type.upper()} Score over all {dataset_type.capitalize()}ing Simulations [m/s]'
    # )

    score_ax.set(xlabel='Turbine', xticklabels=[f'T{t}' for t in system_fi.downstream_turbine_indices])
    score_ax.set_ylabel(f'RMSE\n[m/s]', rotation=0, ha='right', labelpad=15, y=0.8)
    return score_fig


def plot_ts(all_ds_indices, ds_indices, simulation_results, time):
    """
   GP estimate, true value, noisy measurements of
    effective wind speeds of downstream turbines vs.
    time for one dataset
    Returns:

    """
    
    ts_fig, ts_ax = plt.subplots(len(simulation_results), len(ds_indices),
                                 sharex=True, sharey=True)
     
    c1 = '#1f77b4'
    
    min_val = np.infty
    max_val = -np.infty
    for row_idx, (sim_idx, sim_data) in enumerate(simulation_results):
        ts_ax[row_idx, 0].set_ylabel(f'Sim {sim_idx + 1}\n [m/s]', rotation=0, ha='right', labelpad=15.0, y=0.8)
        for col_idx, ds in enumerate(ds_indices):
            ts_ax[0, col_idx].set_title(f'T{ds}')
            ts_ax[-1, col_idx].set(xlabel='Time [s]',
                                   xticks=list(range(0, time[-1] + 600, 600)),
                                    xticklabels=[f'{t}' for t in list(range(0, time[-1] + 600, 600))])
            ds_idx = all_ds_indices.index(ds)
            
            # ts_ax[ds_idx, ax_idx].scatter(time, simulation_results[dataset_type][sim_idx]['true'][:, ds_idx],
            #                    color='orange', label=f'True', marker="o")
            ts_ax[row_idx, col_idx].plot(time, sim_data['pred'][:, ds_idx], color='green', label=f'Predicted Mean')
            # ts_ax[ds_idx, ax_idx].plot(time, simulation_results[dataset_type][sim_idx]['modeled'][:, ds_idx],
            #                    color='purple', label=f'Base Modeled')
            ts_ax[row_idx, col_idx].plot(time, sim_data['meas'][:, ds_idx] - sim_data['modeled'][:, ds_idx],
                                          color='red', label=f'Measurements')
            ts_ax[row_idx, col_idx].fill_between(time, sim_data['pred'][:, ds_idx] - sim_data['std'][:, ds_idx],
                                       sim_data['pred'][:, ds_idx] + sim_data['std'][:, ds_idx],
                                       alpha=0.1, label=f'Predicted Std. Dev.', color='lightgreen')

            min_val = min(min_val, np.nanmin([ln.get_ydata() for ln in ts_ax[row_idx, col_idx].get_lines()]))
            max_val = max(max_val, np.nanmax([ln.get_ydata() for ln in ts_ax[row_idx, col_idx].get_lines()]))
    
    min_val = max(min_val, -WIND_SPEED_RANGE[1] / 2)
    max_val = min(max_val, WIND_SPEED_RANGE[0] / 2)
   
    for row_idx, (sim_idx, sim_data) in enumerate(simulation_results):
        for col_idx, ds in enumerate(ds_indices):
            ds_idx = all_ds_indices.index(ds)
     
            training_start_idx = [time[k_idx] for k_idx, ts in enumerate(sim_data['training_size']) if ts[ds_idx] > 0][0]
            max_training_size = sim_data['max_training_size']
            training_end_idx = [time[k_idx] for k_idx, ts in enumerate(sim_data['training_size']) if ts[ds_idx] == max_training_size][0]
            ts_ax[row_idx, col_idx].plot([training_start_idx, training_start_idx], [min_val, max_val], linestyle='--',
                                       color=c1)
            
            ts_ax[row_idx, col_idx].set(ylim=(min_val, max_val))
            ts_ax[row_idx, col_idx].plot([training_end_idx, training_end_idx], [min_val, max_val], linestyle='--',
                                    color=c1)

            # ts_ax[ax_idx].set(
            #     title=f'Downstream Turbine Effective Wind Speeds for {dataset_type.capitalize()}ing Simulation {j} [m/s]')

    # ts_ax[-1, -1].legend(loc='upper right')
    return ts_fig


def plot_std_evolution(all_ds_indices, ds_indices, simulation_results, time):
    """
    plot evolution of sum of predicted variance at grid test points for middle column downstream turbines vs online training time
    Returns:

    """
    # for each simulation, each time gp.add_training_data is called,
    # the predicted variance is computed for a grid of test points
    # std_fig, std_ax = plt.subplots(len(sim_indices), len(ds_indices), sharex=True, sharey=True)
    std_fig, std_ax = plt.subplots(len(simulation_results), len(ds_indices), sharex=True)
    
    for row_idx, (sim_idx, sim_data) in enumerate(simulation_results):
        std_ax[row_idx, 0].set_ylabel(f'Sim {sim_idx + 1}\n[(m/s)$^2$]', rotation=0, ha='right', labelpad=15.0, y=0.8)
        for col_idx, ds in enumerate(ds_indices):
            std_ax[0, col_idx].set_title(f'T{ds}')
            std_ax[-1, col_idx].set(xlabel='Time [s]',
                                    xticks=list(range(0, time[-1] + 600, 600)),
                                    xticklabels=[f'{t}' for t in list(range(0, time[-1] + 600, 600))])
            ds_idx = all_ds_indices.index(ds)
            
            std_ax[row_idx, col_idx].plot(time, sim_data['test_var'][:, ds_idx], label=f'T{ds}')
    return std_fig


def plot_k_train_evolution(all_ds_indices, ds_indices, simulation_results, time):
    """
    plot evolution of sum of predicted variance at grid test points for middle column downstream turbines vs online training time
    Returns:

    """
    # for each simulation, each time gp.add_training_data is called,
    # the predicted variance is computed for a grid of test points
    k_train_fig, k_train_ax = plt.subplots(len(simulation_results), len(ds_indices), sharex=True, sharey=True)

    for row_idx, (sim_idx, sim_data) in enumerate(simulation_results):
        k_train_ax[row_idx, 0].set_ylabel(f'Sim {sim_idx + 1}\n[s]', rotation=0, ha='right', labelpad=15.0, y=0.8)
        for col_idx, ds in enumerate(ds_indices):
            k_train_ax[0, col_idx].set_title(f'T{ds}')
            k_train_ax[-1, col_idx].set(xlabel='Time [s]', yticks=list(range(0, time[-1] + 600, 600)),
                                        xticks=list(range(0, time[-1] + 600, 600)),
                                    xticklabels=[f'{t}' for t in list(range(0, time[-1] + 600, 600))])
            ds_idx = all_ds_indices.index(ds)
            
            dps = [k_tr[ds_idx] if len(k_tr[ds_idx]) else [np.nan] for k_tr in sim_data['k_train']]
            time_vals = np.concatenate([[time[t_idx]] * len(dp) for t_idx, dp in enumerate(dps)])
            dps = np.concatenate(dps)
            
            k_train_ax[row_idx, col_idx].scatter(time_vals, dps, label=f'T{ds}')

    return k_train_fig

def compute_scores(system_fi, cases, simulation_results, score_type):

    case_vals = []
    sim_vals = []
    turbine_vals = []
    score_vals = []
    # for each downstream turbine
    for i, ds_idx in enumerate(system_fi.downstream_turbine_indices):
        
        # compute the rmse of the turbine effective wind speed error for each simulation
        # turbine_scores = []
        for case_idx, sim_idx, sim in simulation_results:
            y_true = sim['true'][:, i]
            y_pred = sim['pred'][:, i]
            y_modeled = sim['modeled'][:, i]
            if score_type == 'rmse':
                score = np.nanmean(((y_true - y_modeled) - y_pred)**2)**0.5
            elif score_type == 'r2':
                score = 1 - (np.nansum(((y_true - y_modeled) - y_pred)**2) / np.nansum((y_true - np.nanmean(y_true))**2))
            
            case_vals.append(case_idx)
            sim_vals.append(sim_idx)
            turbine_vals.append(ds_idx)
            score_vals.append(score)
        
    scores = pd.DataFrame(data={
        'Case': case_vals,
        'max_training_size': [cases[case_idx]['max_training_size'] for case_idx in case_vals],
        'kernel': [cases[case_idx]['kernel'] for case_idx in case_vals],
        'noise_std': [cases[case_idx]['noise_std'] for case_idx in case_vals],
        'k_delay': [cases[case_idx]['k_delay'] for case_idx in case_vals],
        'batch_size': [cases[case_idx]['batch_size'] for case_idx in case_vals],
        'Simulation': sim_vals,
        'Turbine': turbine_vals,
        f'{score_type}': score_vals
    })

    return scores


def plot_wind_farm(system_fi):
    farm_fig, farm_ax = plt.subplots(1, 1)
    hor_plane = system_fi.get_hor_plane()
    im = visualize_cut_plane(hor_plane, ax=farm_ax)
    divider = make_axes_locatable(farm_ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.15)
    farm_fig.colorbar(im, cax=cax)
    # plot_turbines_with_fi(farm_ax, system_fi)
    for t in system_fi.turbine_indices:
        x = system_fi.layout_x[t] - 100
        y = system_fi.layout_y[t]
        farm_ax.annotate(f'T{t}', (x, y), ha="center", va="center")
    
    farm_ax.set_xlabel('Streamwise Distance [m]')
    farm_ax.set_ylabel('Cross-\nStream\nDistance\n[m]', rotation=0, ha='right', labelpad=15.0, y=0.8)
    
    return farm_fig

def generate_scores_table(scores_df):
    # \begin{tabular}{llllllll}
    # Case & $N_\text
    # {tr}$ & $K(\cdot, \cdot)$ & $\sigma_n$ & $k_\text
    # {delay}$ & $q$ & $RMSE$ \ \
    #     1 &$10$ & SE &$0.01$ & $4$ & $1$ & \ \
    #     2 &$10$ & SE &$0.01$ & $4$ & $2$ & \ \
    #     3 &$10$ & SE &$0.01$ & $4$ & $4$ & \ \
    #     4 &$10$ & SE &$0.01$ & $2$ & $1$ & \ \
    #     5 &$10$ & SE &$0.01$ & $8$ & $1$ & \ \
    #     6 &$10$ & SE &$0.001$ & $4$ & $1$ & \ \
    #     7 &$10$ & SE &$0.1$ & $4$ & $1$ & \ \
    #     8 &$5$ & SE &$0.01$ & $4$ & $1$ & \ \
    #     9 &$20$ & SE &$0.01$ & $4$ & $1$ & \ \
    #     10 &$10$ & Mat\'ern&$0.01$&$4$&$1$&
    # \end{tabular}
    
    captions = [
    '$RMSE$ of the \\ac{GP}-Predicted vs. True Effective Velocities for Different Learning Parameters'
    ]
    
    labels = [
        'tab:scores'
    ]
    project_dir = '/Users/aoifework/Documents/Research/wake_modeling'
    dest_results_dir = os.path.join(project_dir, 'tables')
    if not os.path.exists(dest_results_dir):
        os.mkdir(dest_results_dir)
    
    
    for caption, label in zip(captions, labels):
        
        export_df = pd.DataFrame(scores_df)
        for row_idx, row in enumerate(export_df['kernel']):
            if 'RBF' in row.__str__():
                scores_df['kernel'].iloc[row_idx] = r"SE"
            elif 'Matern' in row.__str__():
                scores_df['kernel'].iloc[row_idx] = r"Mat\'ern"
        
        # colour lowest rmse with lightest gray, highest with darkest gray
        for row_idx, row in enumerate(export_df['rmse']):
            scores_df['rmse'].iloc[row_idx] = r'\cellcolor{Gray' + f'{10 - row_idx}' + r'}' + format(scores_df["rmse"].iloc[row_idx], '.3f')
        
        # Rename features in Candidate Function column
        rename_mapping = {
            'max_training_size': r'$N_\text{tr}$',
            'kernel': r'$K(\cdot, \cdot)$',
            'noise_std': r'$\sigma_n$',
            'k_delay': r'$k_\text{delay}$',
            'batch_size': r'$q$',
            'rmse': r'$RMSE$'
        }
        export_df = scores_df.rename(columns=rename_mapping)
        
        # reorder columns
        cols = [rename_mapping[col] for col in ['max_training_size', 'kernel', 'noise_std', 'k_delay', 'batch_size', 'rmse']]
        export_df = export_df[cols]
        
        # for old, new in rename_mapping.items():
            # scores_df['Candidate Function'] = scores_df['Candidate Function'].str.replace(old, new)
        export_df.reset_index(level=0, inplace=True)
        export_df['Case'] = export_df['Case'] + 1
        export_df.sort_values(by='Case', inplace=True)
        print(export_df)
        
        with open(os.path.join(dest_results_dir, 'case_scores.tex'), 'w') as fp:
            export_df.to_latex(buf=fp,
                             float_format='%.3f',
                             caption=caption, label=label,
                             position='!ht', escape=False, index=False)