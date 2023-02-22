from typing import Dict, Generator, Tuple

import numpy as np
from numpy.polynomial import Polynomial

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase
from lib.Curves.CurveBase import CurveBase
from lib.Curves.CurvePolynomial import CurvePolynomial
from lib.StencilCreators import Stencil
from src.Indexers import ArrayIndexerNd


class ELVIRACurveCellCreator(CurveCellCreatorBase):
    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[CurveBase, None, None]:
        stencil_values = stencil.values.sum(axis=1 - independent_axis)

        point_x = coords[independent_axis] + 0.5
        point_y = coords[1 - independent_axis] - 1.0
        # which side the integral has to be done, is 0 below the curve or is 1?
        try:
            jump = regular_opposite_cells[1].evaluate(coords.coords) - regular_opposite_cells[0].evaluate(coords.coords)
        except:
            regular_opposite_cells[1].evaluate(coords.coords)
        if jump < 0:
            point_y += stencil_values[1]
        else:
            point_y += 3 * jump - stencil_values[1]
            stencil_values = 3 * jump - stencil_values

        for slope in [stencil_values[1] - stencil_values[0], stencil_values[2] - stencil_values[1],
                      (stencil_values[2] - stencil_values[0]) / 2]:
            yield CurvePolynomial(
                polynomial=Polynomial([point_y - slope * point_x, slope]),
                value_up=regular_opposite_cells[1].evaluate(coords.coords),
                value_down=regular_opposite_cells[0].evaluate(coords.coords)
            )
