from typing import Dict, Generator, Tuple

import numpy as np
from numpy.polynomial import Polynomial

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase
from lib.Curves.CurveBase import CurveBase
from lib.Curves.CurvePolynomial import CurvePolynomial
from lib.StencilCreators import Stencil
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd


class ELVIRACurveCellCreator(CurveCellCreatorBase):
    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[CurveBase, None, None]:
        value_up = regular_opposite_cells[1].evaluate(coords.coords)
        value_down = regular_opposite_cells[0].evaluate(coords.coords)
        # if the values are not 0 or 1
        min_value = np.min((value_up, value_down))
        stencil_values = stencil.values.reshape((3, 3)).sum(axis=1 - independent_axis) - 3 * min_value

        point_x = coords[independent_axis] + 0.5
        point_y = coords[1 - independent_axis] - 1.0
        # which side the integral has to be done, is 0 below the curve or is 1?
        jump = value_up - value_down
        if jump < 0:
            point_y += stencil_values[1]
        else:
            point_y += 3 * jump - stencil_values[1]
            stencil_values = 3 * jump - stencil_values

        for slope in [stencil_values[1] - stencil_values[0], stencil_values[2] - stencil_values[1],
                      (stencil_values[2] - stencil_values[0]) / 2]:
            yield CurvePolynomial(
                polynomial=Polynomial([point_y - slope * point_x, slope]),
                value_up=value_up,
                value_down=value_down
            )
