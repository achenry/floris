# Copyright 2021 NREL

# Licensed under the Apache License, Version 2.0 (the "License"); you may not
# use this file except in compliance with the License. You may obtain a copy of
# the License at http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
# WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
# License for the specific language governing permissions and limitations under
# the License.

from typing import Any, Dict

import numexpr as ne
import numpy as np

from ....utilities import cosd, sind, tand
from .gaussian_model_base import GaussianModel
from ..base_velocity_deficit import VelocityDeficit


class EmpiricalGaussVelocityDeficit(VelocityDeficit):
    """
    The Empirical Gauss velocity model has a Gaussian profile
    (see :cite:`bastankhah2016experimental` and
    :cite:`King2019Controls`) throughout and expands in a (smoothed)
    piecewise linear fashion.

    parameter_dictionary (dict): Model-specific parameters.
        Default values are used when a parameter is not included
        in `parameter_dictionary`. Possible key-value pairs include:

            -   **wake_expansion_rates** (*list*): List of expansion
                rates for the Gaussian wake width. Must be of length 1
                or greater.
            -   **breakpoints_D** (*list*): List of downstream
                locations, specified in terms of rotor diameters, where
                the expansion rates go into effect. Must be one element
                shorter than wake_expansion_rates. May be empty.
            -   **sigma_0_D** (*float*): Initial width of the Gaussian
                wake at the turbine location, specified as a multiplier
                of the rotor diameter.
            -   **smoothing_length_D** (*float*): Distance over which
                the corners in the piece-wise linear wake expansion rate
                are smoothed (specified as a multiplier of the rotor
                diameter).
            -   **mixing_gain_deflection** (*float*): Gain to set the
                increase in wake expansion due to wake-induced mixing.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
    """
    default_parameters = {"wake_expansion_rates": [0.023, 0.008],
                          "breakpoints_D": [10],
                          "sigma_0_D": 0.28,
                          "smoothing_length_D": 2.0,
                          "mixing_gain_velocity": 2.0}
    
    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                -   **we** (*float*): The linear wake decay constant that
                    defines the cone boundary for the wake as well as the
                    velocity deficit. D/2 +/- we*x is the cone boundary for the
                    wake.
        """
        super().__init__(parameter_dictionary)
        self.model_string = "empirical_gauss"
        model_dictionary = self._get_model_dict(__class__.default_parameters)
        
        self.wake_expansion_rates = [float(x) for x in model_dictionary["wake_expansion_rates"]]
        self.breakpoints_D = [float(x) for x in model_dictionary["breakpoints_D"]]
        self.sigma_0_D = float(model_dictionary["sigma_0_D"])
        self.smoothing_length_D = float(model_dictionary["smoothing_length_D"])
        self.mixing_gain_velocity = float(model_dictionary["mixing_gain_velocity"])
    
    @property
    def wake_expansion_rates(self):
        return self._wake_expansion_rates
    
    @property
    def breakpoints_D(self):
        return self._breakpoints_D
    
    @property
    def sigma_0_D(self):
        return self._sigma_0_D
    
    @property
    def smoothing_length_D(self):
        return self._smoothing_length_D
    
    @property
    def mixing_gain_velocity(self):
        return self._mixing_gain_velocity
    
    @wake_expansion_rates.setter
    def wake_expansion_rates(self, value):
        if type(value) is not list:
            err_msg = (
                "Invalid value type given for wake_expansion_rates: {}, " + "expected list."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._wake_expansion_rates = value
        if value != __class__.default_parameters["wake_expansion_rates"]:
            self.logger.info(
                (
                    "Current value of wake_expansion_rates, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["wake_expansion_rates"])
            )
    
    @breakpoints_D.setter
    def breakpoints_D(self, value):
        if type(value) is not list:
            err_msg = (
                "Invalid value type given for breakpoints_D: {}, " + "expected list."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._breakpoints_D = value
        if value != __class__.default_parameters["breakpoints_D"]:
            self.logger.info(
                (
                    "Current value of breakpoints_D, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["breakpoints_D"])
            )
    
    @sigma_0_D.setter
    def sigma_0_D(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for sigma_0_D: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._sigma_0_D = value
        if value != __class__.default_parameters["sigma_0_D"]:
            self.logger.info(
                (
                    "Current value of sigma_0_D, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["sigma_0_D"])
            )
    
    @smoothing_length_D.setter
    def smoothing_length_D(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for smoothing_length_D: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._smoothing_length_D = value
        if value != __class__.default_parameters["smoothing_length_D"]:
            self.logger.info(
                (
                    "Current value of smoothing_length_D, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["smoothing_length_D"])
            )
    
    @mixing_gain_velocity.setter
    def mixing_gain_velocity(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for mixing_gain_velocity: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._mixing_gain_velocity = value
        if value != __class__.default_parameters["mixing_gain_velocity"]:
            self.logger.info(
                (
                    "Current value of mixing_gain_velocity, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["mixing_gain_velocity"])
            )
    
    # def prepare_function(
    #     self,
    #     grid: Grid,
    #     flow_field: FlowField,
    # ) -> Dict[str, Any]:
    #
    #     kwargs = {
    #         "x": grid.x_sorted,
    #         "y": grid.y_sorted,
    #         "z": grid.z_sorted,
    #         "wind_veer": flow_field.wind_veer
    #     }
    #     return kwargs

    def function(
        self,
        x_locations,
        y_locations,
        z_locations,
        turbine,
        turbine_coord,
        deflection_field,
        flow_field,
        # axial_induction_i: np.ndarray,
        # deflection_field_y_i: np.ndarray,
        # deflection_field_z_i: np.ndarray,
        # yaw_angle_i: np.ndarray,
        # tilt_angle_i: np.ndarray,
        # mixing_i: np.ndarray,
        # ct_i: np.ndarray,
        # hub_height_i: float,
        # rotor_diameter_i: np.ndarray,
        # enforces the use of the below as keyword arguments and adherence to the
        # unpacking of the results from prepare_function()
        # *,
        # x: np.ndarray,
        # y: np.ndarray,
        # z: np.ndarray,
        # wind_veer: float
    ):
        """
        Calculates the velocity deficits in the wake.

        Args:
            x_i (np.array): Streamwise direction grid coordinates of
                the ith turbine (m).
            y_i (np.array): Cross stream direction grid coordinates of
                the ith turbine (m).
            z_i (np.array): Vertical direction grid coordinates of
                the ith turbine (m) [not used].
            axial_induction_i (np.array): Axial induction factor of the
                ith turbine (-) [not used].
            deflection_field_y_i (np.array): Horizontal wake deflections
                due to the ith turbine's yaw misalignment (m).
            deflection_field_z_i (np.array): Vertical wake deflections
                due to the ith turbine's tilt angle (m).
            yaw_angle_i (np.array): Yaw angle of the ith turbine (deg).
            tilt_angle_i (np.array): Tilt angle of the ith turbine
                (deg).
            mixing_i (np.array): The wake-induced mixing term for the
                ith turbine.
            ct_i (np.array): Thrust coefficient for the ith turbine (-).
            hub_height_i (float): Hub height for the ith turbine (m).
            rotor_diameter_i (np.array): Rotor diameter for the ith
                turbine (m).

            x (np.array): Streamwise direction grid coordinates of the
                flow field domain (m).
            y (np.array): Cross stream direction grid coordinates of the
                flow field domain (m).
            z (np.array): Vertical direction grid coordinates of the
                flow field domain (m).
            wind_veer (np.array): Wind veer (deg).

        Returns:
            np.array: Velocity deficits (-).
        """

        include_mirror_wake = True # Could add this as a user preference.
        
        # veer (degrees)
        veer = flow_field.wind_veer
        
        # added turbulence model
        TI = turbine.current_turbulence_intensity
        
        # turbine parameters
        D = turbine.rotor_diameter
        HH = turbine.hub_height
        yaw_angle = -1 * turbine.yaw_angle  # opposite sign convention in this model
        tilt_angle = turbine.tilt_angle
        Ct = turbine.Ct
        U_local = flow_field.u_initial
        mixing = turbine.mixing
        
        # wake deflection
        delta = deflection_field
        
        xR, _ = GaussianModel.mask_upstream_wake(turbine_coord.x2, turbine_coord, yaw_angle)

        # Initial wake widths
        sigma_y0 = self.sigma_0_D * D * cosd(yaw_angle)
        sigma_z0 = self.sigma_0_D * D * cosd(tilt_angle)

        # No specific near, far wakes in this model
        downstream_mask = (x_locations > turbine_coord.x1 + 0.1)
        upstream_mask = (x_locations < turbine_coord.x1 - 0.1)

        # Wake expansion in the lateral (y) and the vertical (z)
        # TODO: could compute shared components in sigma_z, sigma_y
        # with one function call.
        sigma_y = empirical_gauss_model_wake_width(
            x_locations - turbine_coord.x1,
            self.wake_expansion_rates,
            [b * D for b in self.breakpoints_D], # .flatten()[0]
            sigma_y0,
            self.smoothing_length_D * D,
            self.mixing_gain_velocity * mixing,
        )
        sigma_y[upstream_mask] = \
            np.tile(sigma_y0, np.shape(sigma_y)[2:])[upstream_mask]

        sigma_z = empirical_gauss_model_wake_width(
            x_locations - turbine_coord.x1,
            self.wake_expansion_rates,
            [b * D for b in self.breakpoints_D], # .flatten()[0]
            sigma_z0,
            self.smoothing_length_D * D,
            self.mixing_gain_velocity * mixing,
        )
        sigma_z[upstream_mask] = \
            np.tile(sigma_z0, np.shape(sigma_z)[2:])[upstream_mask]

        # 'Standard' wake component
        r, C = rCalt(
            veer,
            sigma_y,
            sigma_z,
            Ct,
            yaw_angle,
            tilt_angle,
            D,
            sigma_y0,
            sigma_z0
        )
        # Normalize to match end of actuator disk model tube
        C = C / (8 * self.sigma_0_D**2 )

        wake_deficit = GaussianModel.gaussian_function(U_local, C, r, 1, np.sqrt(0.5))

        if include_mirror_wake:
            # TODO: speed up this option by calculating various elements in
            #       rCalt only once.
            # Mirror component
            r_mirr, C_mirr = rCalt(
                veer, # TODO: Is veer OK with mirror wakes?
                sigma_y,
                sigma_z,
                Ct,
                yaw_angle,
                tilt_angle,
                D,
                sigma_y0,
                sigma_z0
            )
            # Normalize to match end of acuator disk model tube
            C_mirr = C_mirr / (8 * self.sigma_0_D**2)

            # ASSUME sum-of-squares superposition for the real and mirror wakes
            wake_deficit = np.sqrt(
                wake_deficit**2 +
                GaussianModel.gaussian_function(U_local, C_mirr, r_mirr, 1, np.sqrt(0.5))**2
            )

        velDef = wake_deficit * downstream_mask

        # return velDef # TODO check if tuple
    
        return velDef, np.zeros(np.shape(velDef)), np.zeros(np.shape(velDef))

def rCalt(wind_veer, sigma_y, sigma_z, Ct,
    yaw, tilt, D, sigma_y0, sigma_z0):

    ## Numexpr
    wind_veer = np.deg2rad(wind_veer)
    a = ne.evaluate(
        "cos(wind_veer) ** 2 / (2 * sigma_y ** 2) + sin(wind_veer) ** 2 / (2 * sigma_z ** 2)"
    )
    b = ne.evaluate(
        "-sin(2 * wind_veer) / (4 * sigma_y ** 2) + sin(2 * wind_veer) / (4 * sigma_z ** 2)"
    )
    c = ne.evaluate(
        "sin(wind_veer) ** 2 / (2 * sigma_y ** 2) + cos(wind_veer) ** 2 / (2 * sigma_z ** 2)"
    )
    r = ne.evaluate(
        "a * ( (y - y_i - delta_y) ** 2) - "+\
        "2 * b * (y - y_i - delta_y) * (z - HH - delta_z) + "+\
        "c * ((z - HH - delta_z) ** 2)"
    )
    d = 1 - Ct * (sigma_y0 * sigma_z0)/(sigma_y * sigma_z) * cosd(yaw) * cosd(tilt)
    C = ne.evaluate("1 - sqrt(d)")
    return r, C

def sigmoid_integral(x, center=0, width=1):
    y = np.zeros_like(x)
    #TODO: Can this be made faster?
    above_smoothing_zone = (x-center) > width/2
    y[above_smoothing_zone] = (x-center)[above_smoothing_zone]
    in_smoothing_zone = ((x-center) >= -width/2) & ((x-center) <= width/2)
    z = ((x-center)/width + 0.5)[in_smoothing_zone]
    if width.shape[0] > 1: # multiple turbine sizes
        width = np.broadcast_to(width, x.shape)[in_smoothing_zone]
    y[in_smoothing_zone] = (width*(z**6 - 3*z**5 + 5/2*z**4)).flatten()
    return y

def empirical_gauss_model_wake_width(
    x,
    wake_expansion_rates,
    breakpoints,
    sigma_0,
    smoothing_length,
    mixing_final,
    ):
    assert len(wake_expansion_rates) == len(breakpoints) + 1, \
        "Invalid combination of wake_expansion_rates and breakpoints."

    sigma = (wake_expansion_rates[0] + mixing_final) * x + sigma_0
    for ib, b in enumerate(breakpoints):
        sigma += (wake_expansion_rates[ib+1] - wake_expansion_rates[ib]) * \
            sigmoid_integral(x, center=b, width=smoothing_length)

    return sigma
