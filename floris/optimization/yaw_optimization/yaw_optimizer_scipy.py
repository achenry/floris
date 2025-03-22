
import numpy as np
from scipy.optimize import minimize
from functools import partial
from concurrent.futures import ProcessPoolExecutor

from .yaw_optimization_base import YawOptimization


class YawOptimizationScipy(YawOptimization):
    """
    YawOptimizationScipy is a subclass of
    :py:class:`floris.optimization.general_library.YawOptimization` that is
    used to optimize the yaw angles of all turbines in a Floris Farm for a single
    set of inflow conditions using the SciPy optimize package.
    """

    def __init__(
        self,
        fmodel,
        minimum_yaw_angle=0.0,
        maximum_yaw_angle=25.0,
        yaw_angles_baseline=None,
        x0=None,
        opt_method="SLSQP",
        opt_options=None,
        turbine_weights=None,
        exclude_downstream_turbines=True,
        verify_convergence=False,
        parallel=False,
        per_wd_sample=False
    ):
        """
        Instantiate YawOptimizationScipy object with a FlorisModel object
        and assign parameter values.
        """
        valid_op_models = ["cosine-loss"]
        if fmodel.get_operation_model() not in valid_op_models:
            raise ValueError(
                "YawOptimizationScipy is currently limited to the following operation models: "
                + ", ".join(valid_op_models)
            )
        if opt_options is None:
            # Default SciPy parameters
            opt_options = {
                "maxiter": 100,
                "disp": True,
                "iprint": 2,
                "ftol": 1e-12,
                "eps": 0.1,
            }

        super().__init__(
            fmodel=fmodel,
            minimum_yaw_angle=minimum_yaw_angle,
            maximum_yaw_angle=maximum_yaw_angle,
            yaw_angles_baseline=yaw_angles_baseline,
            x0=x0,
            turbine_weights=turbine_weights,
            normalize_control_variables=True,
            calc_baseline_power=True,
            exclude_downstream_turbines=exclude_downstream_turbines,
            verify_convergence=verify_convergence,
            per_wd_sample=per_wd_sample
        )

        self.opt_method = opt_method
        self.opt_options = opt_options
        self.parallel = parallel

    def optimize(self):
        """
        Find optimum setting of turbine yaw angles for a single turbine
        cluster that maximizes the weighted wind farm power production
        given fixed atmospheric conditions (wind speed, direction, etc.)
        using the scipy.optimize.minimize function.

        Returns:
            opt_yaw_angles (np.array): Optimal yaw angles in degrees. This
            array is equal in length to the number of turbines in the farm.
        """
        # Loop through every wind condition individually
        wd_array = self.fmodel_subset.core.flow_field.wind_directions
        ws_array = self.fmodel_subset.core.flow_field.wind_speeds
        ti_array = self.fmodel_subset.core.flow_field.turbulence_intensities
        if self.per_wd_sample:
            wd_sample_indices = np.repeat(np.arange(self.fmodel_subset.n_sample_points), (self.fmodel_subset.n_unexpanded,))
            wd_array = np.tile(wd_array, (self.fmodel_subset.n_sample_points,))
            ws_array = np.tile(ws_array, (self.fmodel_subset.n_sample_points,))
            ti_array = np.tile(ti_array, (self.fmodel_subset.n_sample_points,))
            _minimum_yaw_angle_subset_norm = np.tile(self._minimum_yaw_angle_subset_norm, (self.fmodel_subset.n_sample_points,1))
            _maximum_yaw_angle_subset_norm = np.tile(self._maximum_yaw_angle_subset_norm, (self.fmodel_subset.n_sample_points,1))
            _turbs_to_opt_subset = np.tile(self._turbs_to_opt_subset, (self.fmodel_subset.n_sample_points,1))
            _x0_subset_norm = np.tile(self._x0_subset_norm, (self.fmodel_subset.n_sample_points,1))
            _farm_power_baseline_subset = self._farm_power_baseline_subset.T.flatten()
            _yaw_angles_template_subset = np.tile(self._yaw_angles_template_subset, (self.fmodel_subset.n_sample_points,1))
            _turbine_weights_subset = np.tile(self._turbine_weights_subset, (self.fmodel_subset.n_sample_points,1))
        else:
            wd_sample_indices = [None] * wd_array.shape[0]
            _minimum_yaw_angle_subset_norm = _minimum_yaw_angle_subset_norm
            _maximum_yaw_angle_subset_norm = _maximum_yaw_angle_subset_norm
            _turbs_to_opt_subset = _turbs_to_opt_subset
            _x0_subset_norm = _x0_subset_norm
            _farm_power_baseline_subset = _farm_power_baseline_subset
            _yaw_angles_template_subset = _yaw_angles_template_subset
            _turbine_weights_subset = _turbine_weights_subset

        het_sm_arr = np.array(self.fmodel.core.flow_field.heterogeneous_inflow_config['speed_multipliers']) if (hasattr(self.fmodel.core.flow_field, 'heterogeneous_inflow_config') and
                    self.fmodel.core.flow_field.heterogeneous_inflow_config is not None) else None
        if het_sm_arr is not None and self.per_wd_sample:
            het_sm_arr = np.tile(het_sm_arr, (self.fmodel_subset.n_sample_points,))
            
        if self.parallel:
            with ProcessPoolExecutor() as executor:
                futures = [executor.submit(optimize_yaw_angles, fmodel=self.fmodel.copy(), 
                                           wd=wd, ws=ws, ti=ti, wd_sample_idx=wd_sample_idx,
                                            turbs_to_opt=_turbs_to_opt_subset[i, :],
                                            yaw_lb=_minimum_yaw_angle_subset_norm[i, _turbs_to_opt_subset[i, :]],
                                            yaw_ub=_maximum_yaw_angle_subset_norm[i, _turbs_to_opt_subset[i, :]],
                                            x0=_x0_subset_norm[i, _turbs_to_opt_subset[i, :]],
                                            J0=_farm_power_baseline_subset[i],
                                            yaw_template=_yaw_angles_template_subset[i, :],
                                            het_sm=None if het_sm_arr is None else het_sm_arr[i, :].reshape(1, -1),
                                            turbine_weights=_turbine_weights_subset[i, :],
                                            normalization_length=self._normalization_length,
                                            calculate_farm_power_func=self._calculate_farm_power,
                                            opt_method=self.opt_method, opt_options=self.opt_options) 
                           for i, (wd, ws, ti, wd_sample_idx) in enumerate(zip(wd_array, ws_array, ti_array, wd_sample_indices))]
                
                residual_plants = [fut.result() for fut in futures]

        else:
            for i, (wd, ws, ti, wd_sample_idx) in enumerate(zip(wd_array, ws_array, ti_array, wd_sample_indices)):
                # Handle heterogeneous inflow, if there is one
                residual_plants = optimize_yaw_angles(fmodel=self.fmodel.copy(), 
                                    wd=wd, ws=ws, ti=ti, wd_sample_idx=wd_sample_idx,
                                    turbs_to_opt=_turbs_to_opt_subset[i, :],
                                    yaw_lb=_minimum_yaw_angle_subset_norm[i, _turbs_to_opt_subset[i, :]],
                                    yaw_ub=_maximum_yaw_angle_subset_norm[i, _turbs_to_opt_subset[i, :]],
                                    x0=_x0_subset_norm[i, _turbs_to_opt_subset[i, :]],
                                    J0=_farm_power_baseline_subset[i],
                                    yaw_template=_yaw_angles_template_subset[i, :],
                                    het_sm=None if het_sm_arr is None else het_sm_arr[i, :].reshape(1, -1),
                                    turbine_weights=_turbine_weights_subset[i, :],
                                    normalization_length=self._normalization_length,
                                    calculate_farm_power_func=self._calculate_farm_power)

        if self.per_wd_sample and self._yaw_angles_opt_subset.ndim < 3:
            self._yaw_angles_opt_subset = np.tile(self._yaw_angles_opt_subset[:, :, np.newaxis], (1, 1, self.fmodel_subset.n_sample_points))
        # Undo normalization/masks and save results to self
        for i, residual_plant in enumerate(residual_plants):
            if residual_plant is None:
                continue
            J0 = _farm_power_baseline_subset[i]
            turbs_to_opt = _turbs_to_opt_subset[i, :]
            if self.per_wd_sample:
                wd_sample_idx = int(i // self.fmodel_subset.n_unexpanded) 
                input_idx = i % self.fmodel_subset.n_unexpanded
                self._farm_power_opt_subset[input_idx, wd_sample_idx] = -residual_plant.fun * J0
                self._yaw_angles_opt_subset[input_idx, turbs_to_opt, wd_sample_idx] = (
                    residual_plant.x * self._normalization_length
                )
            else:
                self._farm_power_opt_subset[i] = -residual_plant.fun * J0
                self._yaw_angles_opt_subset[i, turbs_to_opt] = (
                    residual_plant.x * self._normalization_length
                )

        # Finalize optimization, i.e., retrieve full solutions
        df_opt = self._finalize()
        return df_opt

def optimize_yaw_angles(fmodel, wd, ws, ti, wd_sample_idx, turbs_to_opt, yaw_lb, yaw_ub, x0, J0, yaw_template, het_sm, turbine_weights,
                        normalization_length, calculate_farm_power_func, opt_method, opt_options):
    fmodel.set(
        wind_directions=[wd],
        wind_speeds=[ws],
        turbulence_intensities=[ti]
    )

    # Find turbines to optimize
    if not any(turbs_to_opt):
        return  # Nothing to do here: no turbines to optimize

    # Extract current optimization problem variables (normalized)
    bnds = [(a, b) for a, b in zip(yaw_lb, yaw_ub)]
    yaw_template = np.tile(yaw_template, (1, 1))
    turbine_weights = np.tile(turbine_weights, (1, 1))

    # Define cost function
    def cost(x):
        x_full = np.array(yaw_template, copy=True)
        x_full[0, turbs_to_opt] = x * normalization_length
        # returns value for each element in wd_sample_points if per_wd_sample == True
        cost = (
            - 1.0 * calculate_farm_power_func(
                yaw_angles=x_full,
                wd_array=[wd],
                ws_array=[ws],
                ti_array=[ti],
                turbine_weights=turbine_weights,
                heterogeneous_speed_multipliers=het_sm
            )[0] / J0
        )
        if wd_sample_idx is not None:
            return cost[wd_sample_idx]
        else:
            return cost

    # Perform optimization
    return minimize(
        fun=cost,
        x0=x0,
        bounds=bnds,
        method=opt_method,
        options=opt_options,
    )