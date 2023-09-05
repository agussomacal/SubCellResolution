from pathlib import Path
from typing import Union, Tuple

import numpy as np

from lib.Curves.CurveVertex import CurveVertexPolynomial, CurveVertexLinearAngle
from lib.SubCellSchemes.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.SubCellSchemes.DataManagers.DatasetsManagers.DatasetsBaseManager import DatasetsBaseManager


class DatasetsManagerVertex(DatasetsBaseManager):
    def __init__(self, path2data: Union[str, Path], N: int, kernel_size: Tuple[int, int], min_val: float,
                 max_val: float, workers=np.Inf, recalculate=False, angle_limits=(0, 2),
                 velocity_range: Tuple[Tuple, Tuple] = ((1e-10, 0), (1.0, 0)), curve_position_radius: float = 1,
                 value_up_random=True):
        """
        params:
            angle_limits: in multiples of pi
        """
        self.angle_limits = angle_limits
        self.curve_position_radius = curve_position_radius
        self.value_up_random = value_up_random
        super().__init__(path2data=path2data, N=N, kernel_size=kernel_size, min_val=min_val, max_val=max_val,
                         recalculate=recalculate, workers=workers, curve=CurveVertexLinearAngle,
                         velocity_range=velocity_range)

    @property
    def base_name(self):
        return f"{super(DatasetsManagerVertex, self).base_name}_" \
               f"AngleLimits{self.angle_limits[0]}pi{self.angle_limits[1]}pi_" \
               f"radius{self.curve_position_radius}{'' if self.value_up_random else '_value_up_fixed'}"

    def get_curve_data(self):
        """
        The center of the stencil is in the center of the kernel_size: //2+0.5
        -> go to DatasetsbaseManager generate_dataset
        """
        angle1, angle2 = np.sort(np.random.uniform(*(np.array(self.angle_limits) * np.pi), size=2))
        angle2 -= angle1
        x0 = np.random.uniform(-self.curve_position_radius, self.curve_position_radius)
        y0 = np.random.uniform(-self.curve_position_radius, self.curve_position_radius)
        value_up = np.random.randint(0, 2) if self.value_up_random else 1
        return angle1, angle2, x0, y0, value_up, 1 - value_up

    # --------- predict/find curve ---------- #
    def create_curve_from_params(self, curve_params, coords: CellCoords, independent_axis: int, value_up, value_down):
        angle1, angle2, x0, y0 = curve_params
        return CurveVertexLinearAngle(angle1, angle2, x0, y0, value_up=value_up, value_down=value_down)
