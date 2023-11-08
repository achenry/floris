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

import numpy as np
from attrs import define, field
from .base_velocity_deflection import VelocityDeflection
from floris.utilities import cosd, sind


class EmpiricalGaussVelocityDeflection(VelocityDeflection):
    """
    The Empirical Gauss deflection model is based on the form of previous the
    Guass deflection model (see :cite:`bastankhah2016experimental` and
    :cite:`King2019Controls`) but simplifies the formulation for simpler
    tuning and more independence from the velocity deficit model.

    parameter_dictionary (dict): Model-specific parameters.
        Default values are used when a parameter is not included
        in `parameter_dictionary`. Possible key-value pairs include:

            -   **horizontal_deflection_gain_D** (*float*): Gain for the
                maximum (y-direction) deflection acheived far downstream
                of a yawed turbine.
            -   **vertical_deflection_gain_D** (*float*): Gain for the
                maximum vertical (z-direction) deflection acheived at a
                far downstream location due to rotor tilt. Specifying as
                -1 will mean that vertical deflections due to tilt match
                horizontal deflections due to yaw.
            -   **deflection_rate** (*float*): Rate at which the
                deflected wake center approaches its maximum deflection.
            -   **mixing_gain_deflection** (*float*): Gain to set the
                reduction in deflection due to wake-induced mixing.
            -   **yaw_added_mixing_gain** (*float*): Sets the
                contribution of turbine yaw misalignment to the mixing
                in that turbine's wake (similar to yaw-added recovery).

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
    """
    default_parameters = {"horizontal_deflection_gain_D": 3.0,
                          "vertical_deflection_gain_D": -1,
                          "deflection_rate": 30,
                          "mixing_gain_deflection": 0.0,
                          "yaw_added_mixing_gain": 0.0,
                          "use_secondary_steering": True,
                          "eps_gain": 0.2}
    
    @property
    def horizontal_deflection_gain_D(self):
        return self._horizontal_deflection_gain_D

    @horizontal_deflection_gain_D.setter
    def horizontal_deflection_gain_D(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for horizontal_deflection_gain_D: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._horizontal_deflection_gain_D = value
        if value != __class__.default_parameters["horizontal_deflection_gain_D"]:
            self.logger.info(
                (
                    "Current value of horizontal_deflection_gain_D, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["horizontal_deflection_gain_D"])
            )
    
    @property
    def vertical_deflection_gain_D(self):
        return self._vertical_deflection_gain_D

    @vertical_deflection_gain_D.setter
    def vertical_deflection_gain_D(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for vertical_deflection_gain_D: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._vertical_deflection_gain_D = value
        if value != __class__.default_parameters["vertical_deflection_gain_D"]:
            self.logger.info(
                (
                    "Current value of vertical_deflection_gain_D, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["vertical_deflection_gain_D"])
            )
    
    @property
    def deflection_rate(self):
        return self._deflection_rate

    @deflection_rate.setter
    def deflection_rate(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for deflection_rate: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._deflection_rate = value
        if value != __class__.default_parameters["deflection_rate"]:
            self.logger.info(
                (
                    "Current value of deflection_rate, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["deflection_rate"])
            )
            
    @property
    def mixing_gain_deflection(self):
        return self._mixing_gain_deflection

    @mixing_gain_deflection.setter
    def mixing_gain_deflection(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for mixing_gain_deflection: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._mixing_gain_deflection = value
        if value != __class__.default_parameters["mixing_gain_deflection"]:
            self.logger.info(
                (
                    "Current value of mixing_gain_deflection, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["mixing_gain_deflection"])
            )
            
    @property
    def yaw_added_mixing_gain(self):
        return self._yaw_added_mixing_gain

    @yaw_added_mixing_gain.setter
    def yaw_added_mixing_gain(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for yaw_added_mixing_gain: {}, " + "expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._yaw_added_mixing_gain = value
        if value != __class__.default_parameters["yaw_added_mixing_gain"]:
            self.logger.info(
                (
                    "Current value of yaw_added_mixing_gain, {0}, is not equal to tuned " + "value of {1}."
                ).format(value, __class__.default_parameters["yaw_added_mixing_gain"])
            )
    
    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                    -   **kd** (*float*): Parameter used to determine the skew
                        angle of the wake.
                    -   **ad** (*float*): Additional tuning parameter to modify
                        the wake deflection with a lateral offset.
                        Defaults to 0.
                    -   **bd** (*float*): Additional tuning parameter to modify
                        the wake deflection with a lateral offset.
                        Defaults to 0.

        """
        super().__init__(parameter_dictionary)
        self.model_string = "empirical_gauss"
        model_dictionary = self._get_model_dict(__class__.default_parameters)
        self.horizontal_deflection_gain_D = float(model_dictionary["horizontal_deflection_gain_D"])
        self.vertical_deflection_gain_D = float(model_dictionary["vertical_deflection_gain_D"])
        self.deflection_rate = float(model_dictionary["deflection_rate"])
        self.mixing_gain_deflection = float(model_dictionary["mixing_gain_deflection"])
        self.yaw_added_mixing_gain = float(model_dictionary["yaw_added_mixing_gain"])
        self.use_secondary_steering = model_dictionary["use_secondary_steering"]
        self.eps_gain = float(model_dictionary["eps_gain"])

    
    def function(
        self, x_locations, y_locations, z_locations, turbine, coord, flow_field
    ):
        """
        Calculates the deflection field of the wake.

        Args:
            x_i (np.array): Streamwise direction grid coordinates of
                the ith turbine (m).
            y_i (np.array): Cross stream direction grid coordinates of
                the ith turbine (m) [not used].
            yaw_i (np.array): Yaw angle of the ith turbine (deg).
            tilt_i (np.array): Tilt angle of the ith turbine (deg).
            mixing_i (np.array): The wake-induced mixing term for the
                ith turbine.
            ct_i (np.array): Thrust coefficient for the ith turbine (-).
            rotor_diameter_i (np.array): Rotor diamter for the ith
                turbine (m).

            x (np.array): Streamwise direction grid coordinates of the
                flow field domain (m).

        Returns:
            np.array: Deflection field for the wake.
        """
        # ==============================================================
        
        # turbine parameters
        D = turbine.rotor_diameter
        yaw_angle = -1 * turbine.yaw_angle  # opposite sign convention in this model
        tilt_angle = turbine.tilt_angle
        Ct = turbine.Ct
        mixing = turbine.mixing

        deflection_gain_y = self.horizontal_deflection_gain_D * D
        if self.vertical_deflection_gain_D == -1:
            deflection_gain_z = deflection_gain_y
        else:
            deflection_gain_z = self.vertical_deflection_gain_D * D

        # Convert to radians, CW yaw for consistency with other models
        yaw_r = np.pi/180 * yaw_angle
        tilt_r = np.pi/180 * tilt_angle

        A_y = (deflection_gain_y * Ct * yaw_r) / (1 + self.mixing_gain_deflection * mixing)
        A_z = (deflection_gain_z * Ct * tilt_r) / (1 + self.mixing_gain_deflection * mixing)

        # Apply downstream mask in the process
        x_normalized = (x_locations - coord.x1) * (x_locations > coord.x1 + 0.1) / D

        log_term = np.log(
            (x_normalized - self.deflection_rate) / (x_normalized + self.deflection_rate)
            + 2
        )

        deflection_y = A_y * log_term
        deflection_z = A_z * log_term

        return deflection_y, deflection_z

def yaw_added_wake_mixing(
    axial_induction_i,
    yaw_angle_i,
    downstream_distance_D_i,
    yaw_added_mixing_gain
):
    return (
        axial_induction_i[:,:,:,0,0]
        * yaw_added_mixing_gain
        * (1 - cosd(yaw_angle_i[:,:,:,0,0]))
        / downstream_distance_D_i**2
    )
