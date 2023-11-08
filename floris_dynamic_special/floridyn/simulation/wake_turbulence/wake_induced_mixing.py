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

from .base_wake_turbulence import WakeTurbulence

class WakeInducedMixing(WakeTurbulence):
    """
    WakeInducedMixing is a model used to generalize wake-added turbulence
    in the Empirical Gaussian wake model. It computes the contribution of each
    turbine to a "wake-induced mixing" term that in turn is used in the
    velocity deficit and deflection models.

    Args:
        parameter_dictionary (dict): Model-specific parameters.
            Default values are used when a parameter is not included
            in `parameter_dictionary`. Possible key-value pairs include:

            -   **atmospheric_ti_gain** (*float*): The contribution of ambient
                turbulent intensity to the wake-induced mixing term. Currently
                throws a warning if nonzero.

    References:
        .. bibliography:: /references.bib
            :style: unsrt
            :filter: docname in docnames
    """
    
    default_parameters = {
        "atmospheric_ti_gain": 0.0
    }
    
    def __init__(self, parameter_dictionary):
        """
        Stores model parameters for use by methods.

        Args:
            parameter_dictionary (dict): Model-specific parameters.
                Default values are used when a parameter is not included
                in `parameter_dictionary`. Possible key-value pairs include:

                -   **initial** (*float*): The initial ambient turbulence
                    intensity, expressed as a decimal fraction.
                -   **constant** (*float*): The constant used to scale the
                    wake-added turbulence intensity.
                -   **ai** (*float*): The axial induction factor exponent used
                    in in the calculation of wake-added turbulence.
                -   **downstream** (*float*): The exponent applied to the
                    distance downstream of an upstream turbine normalized by
                    the rotor diameter used in the calculation of wake-added
                    turbulence.
        """
        super().__init__(parameter_dictionary)
        self.model_string = "wake_induced_mixing"
        model_dictionary = self._get_model_dict(__class__.default_parameters)

        # turbulence parameters
        self.atmospheric_ti_gain = model_dictionary["atmospheric_ti_gain"]
    
    @property
    def atmospheric_ti_gain(self):
        """
		Parameter that is the initial ambient turbulence intensity, expressed as
		a decimal (e.g. 10% TI -> 0.10).

		**Note:** This is a virtual property used to "get" or "set" a value.

		Args:
			ti_initial (float): Initial ambient turbulence intensity.

		Returns:
			float: Initial ambient turbulence intensity.

		Raises:
			ValueError: Invalid value.
		"""
        return self._atmospheric_ti_gain
    
    @atmospheric_ti_gain.setter
    def atmospheric_ti_gain(self, value):
        if type(value) is not float:
            err_msg = (
                "Invalid value type given for " + "atmospheric_ti_gain: {}, expected float."
            ).format(value)
            self.logger.error(err_msg, stack_info=True)
            raise ValueError(err_msg)
        self._atmospheric_ti_gain = value
        if value != __class__.default_parameters["atmospheric_ti_gain"]:
            self.logger.info(
                (
                    "Current value of atmospheric_ti_gain, {0}, is not equal to tuned "
                    + "value of {1}."
                ).format(value, __class__.default_parameters["atmospheric_ti_gain"])
            )

    def function(
        self,
        ambient_TI, coord_ti, turbine_coord, turbine
    ):
        """
        Calculates the contribution of turbine i to all other turbines'
        mixing terms.

        Args:
            axial_induction_i (np.array): Axial induction factor of
                the ith turbine (-).
            downstream_distance_D_i (np.array): The distance downstream
                from turbine i to all other turbines (specified in terms
                of multiples of turbine i's rotor diameter) (D).

        Returns:
            np.array: Components of the wake-induced mixing term due to
                the ith turbine.
        """
        # TODO check if right
        wake_induced_mixing = turbine.aI[:,:,:,0,0] / ((coord_ti.x1 - turbine_coord.x1) / turbine.rotor_diameter)**2

        return wake_induced_mixing
