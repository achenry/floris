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