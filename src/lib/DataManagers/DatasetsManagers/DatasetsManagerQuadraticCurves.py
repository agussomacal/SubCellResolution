from pathlib import Path
from typing import Union, Tuple

import numpy as np
from numpy.linalg import lstsq
from numpy.polynomial import Polynomial

from lib.Curves.CurvePolynomial import CurveQuadraticAngle, CurveQuadratic, INV_VANDER_MAT_3x3
from lib.SubCellSchemes.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.SubCellSchemes.DataManagers.DatasetsManagers.DatasetsBaseManager import DatasetsBaseManager

ANGLE_OBJECTIVE = "angle_objective"
POINTS_OBJECTIVE = "points_objective"
QUADRATIC_PARAMS_OBJECTIVE = "quadratic_params_objective"

# the vander points are in -1.5, 0, 1.5
INV_VANDER_MAT_3x3_3 = np.array(
    [[0, 1, 0],
     [-1 / 3, 0, 1 / 3],
     [2 / 9, -4 / 9, 2 / 9]]
)

POINTS_SAMPLER_EQUISPACE = "equispace"
POINTS_SAMPLER_CHEBYCHEV = "chebychev"
POINTS_SAMPLER_QUADRATIC = "quadratic"


def get_evaluation_points(amplitude, num_points, sampling):
    if sampling == POINTS_SAMPLER_EQUISPACE:
        x = np.linspace(-0.5, 0.5, num=num_points)
    elif sampling == POINTS_SAMPLER_CHEBYCHEV:
        x = np.cos((np.arange(1, num_points + 1) * 2 - 1) / (2 * num_points) * np.pi) / 2
    elif sampling == POINTS_SAMPLER_QUADRATIC:
        x = np.linspace(-1, 1, num=num_points) * np.abs(np.linspace(-0.5, 0.5, num=num_points))
    else:
        raise Exception(f"sampling {sampling} not implemented.")
    x *= amplitude
    vander_mat_extended = np.transpose([np.ones(num_points), x, x ** 2])
    return vander_mat_extended, x


class DatasetsManagerQuadraticCurves(DatasetsBaseManager):
    def __init__(self, path2data: Union[str, Path], N: int, kernel_size: Tuple[int, int], min_val: float,
                 max_val: float, workers=np.Inf, recalculate=False, angle_limits=(0, 2),
                 velocity_range: Tuple[Tuple, Tuple] = ((1e-10, 0), (1.0, 0)),
                 curve_position_radius: Union[float, Tuple] = 1,
                 learning_objective=ANGLE_OBJECTIVE, value_up_random=True, num_points: int = 3,
                 points_sampler: str = POINTS_SAMPLER_EQUISPACE, points_interval_size: float = 1):
        """
        params:
            angle_limits: in multiples of pi
        """
        self.angle_limits = angle_limits
        self.learning_objective = learning_objective
        self.curve_position_radius = curve_position_radius
        self.value_up_random = value_up_random
        self.num_points = num_points
        self.points_sampler = points_sampler
        self.points_interval_size = points_interval_size
        self.vander_mat_extended, self.x = get_evaluation_points(points_interval_size, num_points, points_sampler)
        if learning_objective == ANGLE_OBJECTIVE:
            curve = CurveQuadraticAngle
        elif learning_objective in [POINTS_OBJECTIVE, QUADRATIC_PARAMS_OBJECTIVE]:
            curve = CurveQuadratic
        else:
            raise Exception(f"learning_objective {learning_objective} not implemented.")

        super().__init__(path2data=path2data, N=N, kernel_size=kernel_size, min_val=min_val, max_val=max_val,
                         recalculate=recalculate, workers=workers, curve=curve,
                         velocity_range=velocity_range)

    @property
    def base_name(self):
        return f"{super(DatasetsManagerQuadraticCurves, self).base_name}_" + \
               (f"AngleLimits{self.angle_limits[0]}pi{self.angle_limits[1]}pi_"
                if self.learning_objective == ANGLE_OBJECTIVE else "") + \
               f"radius{self.curve_position_radius}{'' if self.value_up_random else '_value_up_fixed'}"

    @property
    def name4learning(self):
        """In case special data aboit the transformations has to be added."""
        if self.learning_objective == POINTS_OBJECTIVE:
            return f"_{self.points_sampler}_{self.num_points}_{self.points_interval_size}"
        return ""

    def transform_curve_data(self, *args):
        if self.learning_objective == POINTS_OBJECTIVE:
            # gets from the polynomial coefficients the y values on the evaluation points
            return np.transpose([Polynomial(coeffs)(self.x) for coeffs in zip(*args)])
        else:
            return args

    def get_curve_data(self):
        value_up = np.random.randint(0, 2) if self.value_up_random else 1
        if self.learning_objective == ANGLE_OBJECTIVE:
            angle = np.random.uniform(*(np.array(self.angle_limits) * np.pi))
            y0 = np.random.uniform(-self.curve_position_radius, self.curve_position_radius)
            a = np.random.uniform(-self.kernel_size[1], self.kernel_size[1]) * 4
            return angle, y0, a, value_up, 1 - value_up
        else:
            # saves the polynomial coefficients
            if isinstance(self.curve_position_radius, tuple):
                # the x points are in -1.5, 0, 1.5 -> the full stencil.
                cba = INV_VANDER_MAT_3x3_3 @ np.array([np.random.uniform(-ym, ym) for ym in self.curve_position_radius])
            else:
                # default the x points are in -0.5, 0, 0.5 -> the central cell
                cba = INV_VANDER_MAT_3x3 @ np.random.uniform(-self.curve_position_radius, self.curve_position_radius,
                                                             size=3)
            return *cba, value_up, 1 - value_up

    # --------- predict/find curve ---------- #
    def create_curve_from_params(self, curve_params, coords: CellCoords, independent_axis: int, value_up, value_down):
        if self.learning_objective == ANGLE_OBJECTIVE:
            angle, y0, a = curve_params
            return CurveQuadraticAngle(angle, y0, a, value_up=value_up, value_down=value_down)
        else:
            if self.learning_objective == POINTS_OBJECTIVE:
                curve_params = lstsq(self.vander_mat_extended, curve_params, rcond=None)[0]
            return CurveQuadratic(*curve_params, value_up, value_down)
