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
