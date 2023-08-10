from pathlib import Path
from typing import Union, Tuple

import numpy as np

from lib.Curves.CurvePolynomial import CurvePolynomial, CurveLinearAngle
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import DatasetsBaseManager

SLOPE_OBJECTIVE = "slope_objective"
ANGLE_OBJECTIVE = "angle_objective"
COS_SIN_OBJECTIVE = "cos_sin_objective"


class DatasetsManagerLinearCurves(DatasetsBaseManager):
    def __init__(self, path2data: Union[str, Path], N: int, kernel_size: Tuple[int, int], min_val: float,
                 max_val: float, workers=np.Inf, recalculate=False, angle_limits=(0, 2),
                 velocity_range: Tuple[Tuple, Tuple] = ((1e-10, 0), (1.0, 0)), curve_position_radius: float = 1,
                 learning_objective=ANGLE_OBJECTIVE, value_up_random=True):
        """
        params:
            angle_limits: in multiples of pi
        """
        self.angle_limits = angle_limits
        self.learning_objective = learning_objective
        self.curve_position_radius = curve_position_radius
        self.value_up_random = value_up_random
        super().__init__(path2data=path2data, N=N, kernel_size=kernel_size, min_val=min_val, max_val=max_val,
                         recalculate=recalculate, workers=workers, curve=CurveLinearAngle,
                         velocity_range=velocity_range)

    @property
    def base_name(self):
        return f"{super(DatasetsManagerLinearCurves, self).base_name}_" \
               f"AngleLimits{self.angle_limits[0]}pi{self.angle_limits[1]}pi_" \
               f"radius{self.curve_position_radius}{'' if self.value_up_random else '_value_up_fixed'}"

    def transform_curve_data(self, angle, r):
        """
        The data may be saved always in the same format but transformed to get different Learning objectives.
        """

        if self.learning_objective == SLOPE_OBJECTIVE:
            return np.tan(angle), r
        elif self.learning_objective == COS_SIN_OBJECTIVE:
            return np.sin(angle), np.cos(angle), r
        elif self.learning_objective == ANGLE_OBJECTIVE:
            return angle, r
        else:
            raise Exception(f"Learning objective {self.learning_objective} not implemented.")

    def get_curve_data(self):
        angle = np.random.uniform(*(np.array(self.angle_limits) * np.pi))
        r = np.random.uniform(-self.curve_position_radius, self.curve_position_radius)
        value_up = np.random.randint(0, 2) if self.value_up_random else 1
        return angle, r, value_up, 1 - value_up

    # --------- predict/find curve ---------- #
    def create_curve_from_params(self, curve_params, coords: CellCoords, independent_axis: int, value_up, value_down):
        if self.learning_objective == SLOPE_OBJECTIVE:
            slope, y_origin = curve_params
        elif self.learning_objective == COS_SIN_OBJECTIVE:
            sin_angle, cos_angle, y_origin = curve_params
            slope = sin_angle / cos_angle
        elif self.learning_objective == ANGLE_OBJECTIVE:
            angle, y_origin = curve_params
            slope = np.tan(angle)
        else:
            raise Exception(f"Learning objective {self.learning_objective} not implemented.")

        new_y_origin = y_origin + coords.coords[1 - independent_axis] + 0.5 - slope * (
                coords.coords[independent_axis] + 0.5)
        return CurvePolynomial(polynomial=[new_y_origin, slope], value_up=value_up, value_down=value_down)