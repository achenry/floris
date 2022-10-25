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
