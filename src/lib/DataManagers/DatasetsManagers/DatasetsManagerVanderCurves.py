from pathlib import Path
from typing import Union, Tuple, Type

import numpy as np
from numpy.linalg import lstsq
from numpy.polynomial import Polynomial
from tqdm import tqdm

from PerplexityLab.miscellaneous import NamedPartial, clean_str4saving, get_map_function
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.Curves.CurvePolynomial import CurveQuadraticAngle, CurveQuadratic

from lib.Curves.VanderCurves import CurveVander
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import DatasetsBaseManager
from lib.StencilCreators import Stencil

ANGLE_OBJECTIVE = "angle_objective"
POINTS_OBJECTIVE = "points_objective"
PARAMS_OBJECTIVE = "params_objective"
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


class DatasetsManagerVanderCurves(DatasetsBaseManager):
    def __init__(self, path2data: Union[str, Path], N: int, kernel_size: Tuple[int, int], min_val: float,
                 max_val: float, curve_type: Type[CurveVander], workers=np.Inf, recalculate=False,
                 velocity_range: Tuple[Tuple, Tuple] = ((1e-10, 0), (1.0, 0)), learning_objective=POINTS_OBJECTIVE,
                 curve_position_radius: Union[float, Tuple] = 1, value_up_random=True, num_points: int = 3,
                 points_sampler: str = POINTS_SAMPLER_EQUISPACE, points_interval_size: float = 1):
        self.curve_position_radius = curve_position_radius
        self.num_points = num_points
        self.points_sampler = points_sampler
        self.points_interval_size = points_interval_size
        self.vander_mat_extended, self.x_points = get_evaluation_points(points_interval_size, num_points,
                                                                        points_sampler)
        self.learning_objective = learning_objective

        super().__init__(path2data=path2data, N=N, kernel_size=kernel_size, min_val=min_val, max_val=max_val,
                         recalculate=recalculate, workers=workers, curve_type=curve_type,
                         velocity_range=velocity_range, value_up_random=value_up_random)

    def get_curve(self, curve_data, **kwargs):
        return self.curve_type(x_points=self.x_points, y_points=np.array(curve_data), **kwargs)

    @property
    def base_name(self):
        return f"{super(DatasetsManagerVanderCurves, self).base_name}" + \
            f"_radius{clean_str4saving(str(self.curve_position_radius))}{'' if self.value_up_random else '_value_up_fixed'}"

    @property
    def name4learning(self):
        """In case special data aboit the transformations has to be added."""
        return f"_{self.learning_objective}_{self.points_sampler}_{self.num_points}_{self.points_interval_size}"

    def transform_curve_data(self, *args):
        if self.learning_objective == POINTS_OBJECTIVE:
            # gets from the polynomial coefficients the y values on the evaluation points
            return args
        elif self.learning_objective == PARAMS_OBJECTIVE:

            def par_func(y_points):
                curve = self.curve_type(x_points=self.x_points, y_points=y_points)
                return super(CurveVander, curve.__class__).params.fget(curve)

            map_func = get_map_function(self.workers)
            transformed_params = [line for line in tqdm(map_func(par_func, zip(*args)),
                                                        desc=f"Converting data to {self.learning_objective}")]
            return np.transpose(transformed_params)
        else:
            raise Exception(f"Not implemented learning_objective {self.learning_objective}")

    def get_curve_data(self):
        # saves the polynomial coefficients
        if isinstance(self.curve_position_radius, tuple):
            # the x points are in -1.5, 0, 1.5 -> the full stencil.
            y_points = np.array([np.random.uniform(-ym, ym) for ym in self.curve_position_radius])
        else:
            # default the x points are in -0.5, 0, 0.5 -> the central cell
            y_points = np.random.uniform(-self.curve_position_radius, self.curve_position_radius,
                                         size=len(self.x_points))
        return y_points

    # --------- predict/find curve ---------- #
    def create_curve_from_params(self, curve_params, coords: CellCoords, independent_axis: int, value_up, value_down,
                                 stencil: Stencil):

        if self.learning_objective == POINTS_OBJECTIVE:
            curve = self.curve_type(x_points=self.x_points, y_points=curve_params, value_up=value_up,
                                    value_down=value_down)
        elif self.learning_objective == PARAMS_OBJECTIVE:
            # TODO: instead of assigning 2 choose automatically the dependency that is not CurveReparametrization
            curve = self.curve_type.__bases__[2](curve_params, value_down=value_down, value_up=value_up)
        else:
            raise Exception(f"Not implemented learning_objective {self.learning_objective}")
        return curve
