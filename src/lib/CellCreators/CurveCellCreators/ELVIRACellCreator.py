from typing import Dict, Generator, Tuple

import numpy as np
from numpy.polynomial import Polynomial

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase, \
    prepare_stencil4one_dimensionalization
from lib.Curves.CurvePolynomial import CurvePolynomial
from lib.Curves.Curves import Curve
from lib.StencilCreators import Stencil


class ELVIRACurveCellCreator(CurveCellCreatorBase):
    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up, value_down = self.updown_value_getter(coords, regular_opposite_cells)
        stencil_values = prepare_stencil4one_dimensionalization(
            value_up=value_up, value_down=value_down, independent_axis=independent_axis, stencil=stencil,
            smoothness_index=smoothness_index, indexer=indexer)
        stencil_values = stencil_values.sum(axis=1)

        for slope in [stencil_values[1] - stencil_values[0], stencil_values[2] - stencil_values[1],
                      (stencil_values[2] - stencil_values[0]) / 2]:
            curve = CurvePolynomial(
                polynomial=Polynomial([stencil_values[1], slope]),
                value_up=value_up,
                value_down=value_down
            )
            curve.set_x_shift(np.mean(stencil.coords[:, independent_axis]) + 0.5)
            curve.set_y_shift(np.min(stencil.coords[:, 1 - independent_axis]))
            yield curve
