# gprs[gp_idx].X_train = np.vstack([tup[0] for tup in max_variances])
available_training_data = np.vstack(gprs[gp_idx].X_train, X_train_new)

# loop through each training data point and find sum of posterior variance over region between two closest training data points normalized by window-length around data point
# old_variance = []
# for i in np.arange(len(gprs[gp_idx].X_train)):
#     x = gprs[gp_idx].X_train[i, :]
#
#
# # loop through each candidate new training data point and find predicted posterior variance
#
# # select max_training_size datapoints from available_training_data that minimizes integral of variance over full domain
#
# # choose set of datapoints with greatest spread
# # nCr = 110C100
#
# # for each candidate new training data point
# for i in X_train_new:
#     x = X_train_new[i]
#
#     # compute predicted variance
#     _, _, y_std_k = gprs[gp_idx].predict(x, system_fi)



# add and normalize new training data
# for row_idx, row in online_measurements_df.iterrows():
# gprs[gp_idx].add_data(online_measurements_df, system_fi)

##

def add_data(self, measurements_df, system_fi, k_delay=GP_CONSTANTS['K_DELAY'], dt=GP_CONSTANTS['DT']):
	new_current_rows = measurements_df.loc[(max(self.k_added) if len(self.k_added) else -1) + 1:]
	effective_dk = self.compute_effective_dk(system_fi, new_current_rows, k_delay=k_delay, dt=dt)

	# for each current time-step, that has not been added, which has enought historic inputs behind it, in measurements_df, add the new training inputs
	k_to_add, = np.where(new_current_rows.index >= (k_delay * effective_dk))

	if len(k_to_add) == 0:
		return

	X = self.X_scalar.transform(
		[generate_input_vector(measurements_df, k, self.upstream_turbine_indices, effective_dk.loc[k], k_delay)[1] for k
		 in k_to_add])
	y = self.y_scalar.transform(
		measurements_df.loc[k_to_add, f'TurbineWindSpeeds_{self.turbine_index}'].to_numpy().reshape(-1, self.n_outputs))

	self.k_added = self.k_added + list(k_to_add)

	# flag this time-step of training data as being added
	# measurements_df.loc[k_to_add, 'IsAdded'] = True

	# drop the historic inputs we no longer need for ANY downstream turbine
	# measurements_df.drop(measurements_df.loc[measurements_df['IsAdded']], inplace=True)

	self._add_data(X, y)

# def compute_test_r2(self):
#     test_rmse = []
#     for X_test in self.test_points:
#         y_pred, _ = self.gpr.predict(X_test, return_std=True)
#         np.sqrt(np.nanmean(np.square(np.subtract(y_true, y_pred))))
#         test_rmse.append(rmse)
#     return test_rmse

# TODO process dfs into 9 * X_train, y_train for each turbine, shuffle, then select first nrows in intialize
# def generate_training_data_df(gprs, partial_offline_measurements_df, system_fi, k_delay, noise_std, max_training_size):
#     noisy_measurements_df = add_gaussian_noise(system_fi, partial_offline_measurements_df, std=noise_std)
#
#     for gp in gprs:
#
#         k_to_add, effective_dk, reduced_effective_dk, history_exists = \
#             gp.check_history(noisy_measurements_df, system_fi, k_delay=k_delay, dt=GP_CONSTANTS['DT'])
#
#         if not history_exists:
#             continue
#
#         # shuffle order to give random data points to GP, then truncate to number of data points needed
#         shuffle_idx = list(k_to_add.index)
#         np.random.shuffle(shuffle_idx)
#         shuffle_idx = shuffle_idx[:max_training_size]
#
#         # compute the base model values for the wake, given all of the freestream wind speeds, yaw angles and axial induction factors over time for each time-series dataset
#
#         if GP_CONSTANTS['MODEL_TYPE'] == 'error':
#             # initialize to steady-state
#             model_fi.floris.farm.flow_field.mean_wind_speed = noisy_measurements_df.loc[
#                 0, 'FreestreamWindSpeed']
#             model_fi.reinitialize_flow_field(
#                 wind_speed=noisy_measurements_df.loc[0, 'FreestreamWindSpeed'],
#                 wind_direction=noisy_measurements_df.loc[0, 'FreestreamWindDir'])
#             model_fi.calculate_wake(
#                 yaw_angles=[noisy_measurements_df.loc[0, f'YawAngles_{t_idx}']
#                             for t_idx in model_fi.turbine_indices],
#                 axial_induction=[noisy_measurements_df.loc[0, f'AxIndFactors_{t_idx}']
#                                  for t_idx in model_fi.turbine_indices])
#
#             y_modeled = []
#             for k in noisy_measurements_df.index:
#                 sim_time = noisy_measurements_df.loc[k, 'Time']
#                 model_fi.floris.farm.flow_field.mean_wind_speed = noisy_measurements_df.loc[
#                     k, 'FreestreamWindSpeed']
#                 model_fi.reinitialize_flow_field(
#                     wind_speed=noisy_measurements_df.loc[k, 'FreestreamWindSpeed'],
#                     wind_direction=noisy_measurements_df.loc[k, 'FreestreamWindDir'],
#                     sim_time=sim_time)
#                 model_fi.calculate_wake(
#                     yaw_angles=[noisy_measurements_df.loc[k, f'YawAngles_{t_idx}']
#                                 for t_idx in model_fi.turbine_indices],
#                     axial_induction=[noisy_measurements_df.loc[k, f'AxIndFactors_{t_idx}']
#                                      for t_idx in model_fi.turbine_indices],
#                     sim_time=sim_time)
#                 y_modeled.append(model_fi.floris.farm.turbines[gp.turbine_index].average_velocity)
#             y_modeled = np.array(y_modeled)
#             noisy_measurements_df[f'TurbineWindSpeeds_{gp.turbine_index}'].to_numpy()
#         else:
#             y_modeled = np.zeros(len(noisy_measurements_df.index))
#
#         X_train_new, y_train_new, input_labels_new = gp.prepare_data(noisy_measurements_df, k_to_add.loc[shuffle_idx],
#                                                    reduced_effective_dk.loc[shuffle_idx],
#                                                    y_modeled=y_modeled[shuffle_idx],
#                                                    k_delay=k_delay)
#
#         training_data_df = pd.DataFrame(columns=input_labels_new + [f'TurbineWindSpeeds_{gp.turbine_index}'],
#                      data=np.hstack([X_train_new, y_train_new]))
#         training_data_df.to_csv(os.path.join(SAVE_DIR, f'offline_training_data_{dataset_type}_turbine-{}_df-{simulation_idx}'))

## TODO
# def evaluate_gprs(gprs, system_fi, X_norm, y_norm, input_labels, fig_dir, plot_data=GP_CONSTANTS['PLOT_DATA'], k_delay=GP_CONSTANTS['K_DELAY']):
#     for t_idx, gpr in enumerate(gprs):
#         for dataset_type in ['train', 'test']:
#             print(f'Score on {dataset_type} Data:',
#                    gpr.score(X_norm[dataset_type], y_norm[dataset_type][:, t_idx]), sep=' ') # test on training/testing data
#
#         # PLOT TRAINING/TEST DATA AND GP PREDICTION vs AX IND FACTOR INPUT
#         if plot_data:
#             plotting_us_turbine_index = system_fi.upstream_turbine_indices[0]
#             input_type = f'AxIndFactors_{plotting_us_turbine_index}'
#             inputs = [f'{input_type}_minus{i}' for i in [0, int(k_delay // 2), k_delay]]
#             _, ax = plt.subplots(len(inputs), 2)
#             plot_prediction_vs_input(ax[:, 0], gpr, inputs, input_labels, X_norm['train'], y_norm['train'], gpr.X_scalar, gpr.y_scalar, plotting_us_turbine_index, 'train')
#             plot_prediction_vs_input(ax[:, 1], gpr, inputs, input_labels, X_norm['test'], y_norm['test'], gpr.X_scalar, gpr.y_scalar, plotting_us_turbine_index, 'test')
#             plt.show()
#             plt.savefig(os.path.join(fig_dir, 'training_test_prediction.png'))

# def refit_gprs(gprs, X_norm, y_norm):
#     for t_idx, gpr in enumerate(gprs):
#         gpr.add_data(X_norm['train'], y_norm['train'][:, t_idx])
#         gpr.fit()
#     return gprs

# def test_gpr(gpr, system_fi, simulation_case_idx, floris_dir, csv_paths=None, wake_field_dfs=None, model_fi=None,
#              plot_data=GP_CONSTANTS['PLOT_DATA'], collect_raw_data=GP_CONSTANTS['COLLECT_RAW_DATA'],
#              model_type=GP_CONSTANTS['MODEL_TYPE'], k_delay=GP_CONSTANTS['K_DELAY'], dt=GP_CONSTANTS['K_DELAY']):
#     ## MODEL WAKE-FIELD DATA AT TURBINES AND COMPARE TO GP PREDICTIONS
#     if collect_raw_data:
#         simulation_df = wake_field_dfs[simulation_case_idx]
#     else:
#         simulation_df = read_datasets(simulation_case_idx, csv_paths)
#
#     time, X_ts, y_ts = gpr.simulate(simulation_df, system_fi, model_fi=model_fi, floris_dir=floris_dir, model_type=model_type, k_delay=k_delay, dt=dt)
#
#     if plot_data:
#         _, ax = plt.subplots(4, 1, sharex=True)
#         plot_prediction_vs_time(ax, time, X_ts, y_ts)
#         plt.show()
#         plt.savefig(os.path.join(FIG_DIR, 'true_vs_predicted_sim.png'))

# def get_data(measurements_dfs, system_fi, data_dir, k_delay=GP_CONSTANTS['K_DELAY'],
#              collect_raw_data_bool=GP_CONSTANTS['COLLECT_RAW_DATA'], plot_data_bool=GP_CONSTANTS['PLOT_DATA']):
#
#     current_input_labels = ['Time'] + ['FreestreamWindSpeed'] + ['FreestreamWindDir'] \
#                            + [f'TurbineWindSpeeds_{t}' for t in system_fi.turbine_indices] \
#                            + [f'AxIndFactors_{t}' for t in system_fi.turbine_indices] \
#                            + [f'YawAngles_{t}' for t in system_fi.turbine_indices]
#
#     if collect_raw_data_bool:
#
#         ## INSPECT DATA
#         if plot_data_bool:
#             plotting_df_idx = [0]
#             plotting_dfs = [measurements_dfs['train'][df_idx] for df_idx in plotting_df_idx]
#             plotting_ds_turbine_index = system_fi.downstream_turbine_indices[0]
#             plotting_us_turbine_index = system_fi.upstream_turbine_indices[0]
#             # 'FreestreamWindSpeed',
#             input_types = ['FreestreamWindDir',
#                            f'TurbineWindSpeeds_{plotting_us_turbine_index}',
#                            f'AxIndFactors_{plotting_us_turbine_index}',
#                            f'YawAngles_{plotting_us_turbine_index}',
#                            f'TurbineWindSpeeds_{plotting_ds_turbine_index}',
#                            f'TurbineWindDirs_{plotting_ds_turbine_index}']
#
#             _, ax = plt.subplots(len(input_types), 1, sharex=True)
#             plot_raw_measurements_vs_time(ax, plotting_dfs, input_types)
#             plt.show()
#     else:
#         mmry_dict = {}
#         for var in ['time', 'X', 'y', 'measurements_dfs']:
#             with open(os.path.join(data_dir, var), 'rb') as f:
#                 # exec("nonlocal " + var)
#                 # sys._getframe(1).f_locals[var] = pickle.load(f)
#                 # ctypes.pythonapi.PyFrame_LocalsToFast(ctypes.py_object(sys._getframe(1)), ctypes.c_int(0))
#                 # exec(var + " = pickle.load(f)")
#                 mmry_dict[var] = pickle.load(f)
#
#             # locals()[var] = mmry_dict[var]
#         time = mmry_dict['time']
#         X = mmry_dict['X']
#         y = mmry_dict['y']
#         measurements_dfs = mmry_dict['measurements_dfs']
#
#     input_labels = generate_input_labels(system_fi.upstream_turbine_indices, k_delay)
#
#     ## INSPECT DATA
#     if plot_data_bool:
#         # fig = plt.figure(constrained_layout=True)
#
#         # gs = GridSpec(3, 3, figure=fig)
#         delay_indices = [0, int(GP_CONSTANTS['K_DELAY'] // 2)]
#         input_types = [f'AxIndFactors_{plotting_us_turbine_index}', f'YawAngles_{plotting_us_turbine_index}',
#                        f'TurbineWindSpeeds_{plotting_us_turbine_index}']
#         output_labels = [f'Turbine {plotting_ds_turbine_index} Wind Speed [m/s]']
#         fig, axs = plt.subplots(len(delay_indices) + 1 + int(len(output_labels) // len(input_types)), len(input_types), sharex=True)
#         plot_measurements_vs_time(axs, time['train'], X['train'], y['train'], input_labels, input_types, delay_indices, output_labels)
#         plt.show()
#
#     return time, X, y, current_input_labels, input_labels

# def get_domain_window(self, x, window_size=0.1, n_datapoints=10):
    #     x_window = []
    #     X_train_width = np.linalg.norm(np.max(self.X_train, axis=0) - np.min(self.X_train, axis=0))
    #     window_size = X_train_width / self.X_train.shape[0]
    #     # for each data point to add to window
    #     for i in range(n_datapoints):
    #         # choose a random vector direction and length (<= window size)
    #         dir = np.random.uniform(0, 2*np.pi, self.n_inputs)
    #         mag = np.random.uniform(0, window_size, 1)
    #         # add direction cosines
    #         x_dash = x + mag * np.cos(dir)
    #         x_window.append(x_dash)
    #
    #     return x_window
    #
    # def simulate(self, floris_dir, simulation_df, system_fi, model_fi=None,
    #              model_type=GP_CONSTANTS['MODEL_TYPE']):
    #     """_summary_
    #     for given time-series of freestream-wind speed and open-loop axIndFactors and YawAngles, simulate the true vs. GP-predicted effective wind speed at this downstream turbine
    #
    #     Args:
    #         simulation_df (_type_): _description_
    #         system_fi (_type_): _description_
    #         model_fi (_type_, optional): _description_. Defaults to None.
    #         floris_dir (_type_, optional): _description_. Defaults to GP_CONSTANTS['FLORIS_DIR'].
    #         model_type (_type_, optional): _description_. Defaults to GP_CONSTANTS['MODEL_TYPE'].
    #         k_delay (_type_, optional): _description_. Defaults to GP_CONSTANTS['K_DELAY'].
    #         dt (_type_, optional): _description_. Defaults to GP_CONSTANTS['K_DELAY'].
    #
    #     Returns:
    #         _type_: _description_
    #     """
    #
    #     y_true = []
    #     y_pred = []
    #     y_std = []
    #     freestream_wind_speeds = simulation_df['FreestreamWindSpeed'].to_numpy()
    #     ax_ind_factors = np.hstack([simulation_df[f'AxIndFactors_{t_idx}'].to_numpy()[np.newaxis, :] for t_idx in self.upstream_turbine_indices])
    #     yaw_angles = np.hstack([simulation_df[f'YawAngles_{t_idx}'].to_numpy()[np.newaxis, :] for t_idx in self.upstream_turbine_indices])
    #     time = simulation_df['Time']
    #
    #     sim_fi = DynFlorisInterface(floris_dir)
    #
    #     for k, sim_time in enumerate(simulation_df['Time']):
    #
    #         sim_fi.floris.farm.flow_field.mean_wind_speed = simulation_df.loc[k, 'FreestreamWindSpeed']
    #
    #         if model_type == 'error':
    #             model_fi.floris.farm.flow_field.mean_wind_speed = simulation_df.loc[k, 'FreestreamWindSpeed']
    #
    #         y_true_k, y_pred_k, y_std_k = self.predict(simulation_df, sim_time, system_fi, model_fi)
    #
    #         if not np.isnan(y_true_k):
    #             y_true.append(y_true_k)
    #             y_pred.append(y_pred_k)
    #             y_std.append(y_std_k)
    #
    #     y_pred = np.concatenate(y_pred).squeeze()
    #     y_std = np.concatenate(y_std).squeeze()
    #
    #     X_ts = {'Freestream Wind Speed [m/s]': freestream_wind_speeds,
    #             'Axial Induction Factors [-]': ax_ind_factors,
    #             'Yaw Angles [deg]': yaw_angles}
    #     y_ts = {f'Turbine {self.turbine_index} Wind Speed [m/s]': {
    #                 'y_true': y_true,
    #                 'y_pred': y_pred,
    #                 'y_std': y_std
    #                 }
    #             }
    #
    #     return time, X_ts, y_ts