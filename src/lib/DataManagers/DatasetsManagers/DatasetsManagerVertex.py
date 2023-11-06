from pathlib import Path
from typing import Union, Tuple, List

import numpy as np

from PerplexityLab.miscellaneous import clean_str4saving
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.VertexCellCreators.VertexCellCreatorBase import VertexLinearExtended
from lib.Curves.CurveVertex import CurveVertexLinearAngle, CurveVertexLinearAngleAngle, CurveVertexPolynomial
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import DatasetsBaseManager
from lib.StencilCreators import Stencil

CURVE_VERTEX_LINEAR_ANGLE = "CurveVertexLinearAngle"


class DatasetsManagerVertex(DatasetsBaseManager):
    def __init__(self, path2data: Union[str, Path], N: int, kernel_size: Tuple[int, int], min_val: float,
                 max_val: float, workers=np.Inf, recalculate=False, angle1_limits=(0, 2), angle_cone_limits=(0, 2),
                 transpose=False, learning_objective=CURVE_VERTEX_LINEAR_ANGLE,
                 velocity_range: Union[Tuple[Tuple, Tuple], List] = ((1e-10, 0), (1.0, 0)),
                 curve_position_radius: float = 1,
                 value_up_random=True):
        """
        params:
            angle_limits: in multiples of pi
        """
        self.angle1_limits = angle1_limits
        self.angle_cone_limits = angle_cone_limits
        self.curve_position_radius = curve_position_radius
        self.value_up_random = value_up_random
        self.learning_objective = learning_objective
        super().__init__(path2data=path2data, N=N, kernel_size=kernel_size, min_val=min_val, max_val=max_val,
                         recalculate=recalculate, workers=workers, curve_type=CurveVertexLinearAngle,
                         velocity_range=velocity_range, transpose=transpose)

    @property
    def base_name(self):
        return f"{super(DatasetsManagerVertex, self).base_name}_" \
               f"Angle1Limits{self.angle1_limits[0]}pi{self.angle1_limits[1]}pi_" \
               f"AngleConeLimits{self.angle_cone_limits[0]}pi{self.angle_cone_limits[1]}pi_" \
               f"_radius{clean_str4saving(str(self.curve_position_radius))}{'' if self.value_up_random else '_value_up_fixed'}"

    def transform_curve_data(self, angle1, angle2, x0, y0):
        """
        The data may be saved always in the same format but transformed to get different Learning objectives.
        """

        if self.learning_objective == CURVE_VERTEX_LINEAR_ANGLE:
            return angle1, angle2, x0, y0
        else:
            raise Exception(f"Learning objective {self.learning_objective} not implemented.")

    def get_curve_data(self):
        """
        The center of the stencil is in the center of the kernel_size: //2+0.5
        -> go to DatasetsbaseManager generate_dataset
        """
        angle1 = np.random.uniform(*(np.array(self.angle1_limits) * np.pi))
        angle2 = np.random.uniform(*(np.array(self.angle_cone_limits) * np.pi))
        x0 = np.random.uniform(-self.curve_position_radius, self.curve_position_radius)
        y0 = np.random.uniform(-self.curve_position_radius, self.curve_position_radius)
        return angle1, angle2, x0, y0

    # --------- predict/find curve ---------- #
    def create_curve_from_params(self, curve_params, coords: CellCoords, independent_axis: int, value_up, value_down,
                                 stencil: Stencil):
        angle1, angle2, x0, y0 = curve_params
        return CurveVertexLinearAngle(angle1, angle2, x0, y0, value_up=value_up, value_down=value_down)

    def get_params_from_curve(self, curve):
        if self.learning_objective == CURVE_VERTEX_LINEAR_ANGLE:
            y0, slope1, p2slope, x0 = curve.params
            angle1 = np.arctan(slope1)
            angle2 = np.arctan(p2slope) - angle1
            params = angle1, angle2, x0, y0
        else:
            raise Exception("Not implemented.")
        return params


class DatasetsManagerVertexAngleAngle(DatasetsBaseManager):
    def __init__(self, path2data: Union[str, Path], N: int, kernel_size: Tuple[int, int], min_val: float,
                 max_val: float, workers=np.Inf, recalculate=False, angle1_limits=(0, 2), angle2_limits=(0, 2),
                 transpose=False,
                 velocity_range: Union[Tuple[Tuple, Tuple], List] = ((1e-10, 0), (1.0, 0)),
                 curve_position_radius: float = 1,
                 value_up_random=True):
        """
        params:
            angle_limits: in multiples of pi
        """
        self.angle1_limits = angle1_limits
        self.angle2_limits = angle2_limits
        self.curve_position_radius = curve_position_radius
        self.value_up_random = value_up_random
        super().__init__(path2data=path2data, N=N, kernel_size=kernel_size, min_val=min_val, max_val=max_val,
                         recalculate=recalculate, workers=workers, curve_type=CurveVertexLinearAngleAngle,
                         velocity_range=velocity_range, transpose=transpose)

    @property
    def base_name(self):
        return f"{super(DatasetsManagerVertexAngleAngle, self).base_name}_" \
               f"Angle1Limits{self.angle1_limits[0]}pi{self.angle1_limits[1]}pi_" \
               f"Angle2Limits{self.angle2_limits[0]}pi{self.angle2_limits[1]}pi_" \
               f"_radius{clean_str4saving(str(self.curve_position_radius))}{'' if self.value_up_random else '_value_up_fixed'}"

    def get_curve_data(self):
        """
        The center of the stencil is in the center of the kernel_size: //2+0.5
        -> go to DatasetsbaseManager generate_dataset
        """
        angle1 = np.random.uniform(*(np.array(self.angle1_limits) * np.pi))
        angle2 = np.random.uniform(*(np.array(self.angle2_limits) * np.pi))
        x0 = np.random.uniform(-self.curve_position_radius, self.curve_position_radius)
        y0 = np.random.uniform(-self.curve_position_radius, self.curve_position_radius)
        return angle1, angle2, x0, y0

    # --------- predict/find curve ---------- #
    def create_curve_from_params(self, curve_params, coords: CellCoords, independent_axis: int, value_up,
                                 value_down,
                                 stencil: Stencil):
        angle1, angle2, x0, y0 = curve_params
        return CurveVertexLinearAngleAngle(angle1, angle2, x0, y0, value_up=value_up, value_down=value_down)
