from typing import Dict, Generator, Tuple, Callable
import numpy as np
from numpy.polynomial import Polynomial

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase, map2unidimensional
from lib.Curves.CurveBase import CurveBase
from lib.Curves.CurvePolynomial import CurvePolynomial
from lib.StencilCreators import Stencil
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd


class PolynomialCurveCellCreator(CurveCellCreatorBase):

    def __init__(self, degree, regular_opposite_cell_searcher: Callable):
        super().__init__(regular_opposite_cell_searcher)
        self.degree = degree

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[CurveBase, None, None]:
        value_up = regular_opposite_cells[1].evaluate(coords.coords)
        value_down = regular_opposite_cells[0].evaluate(coords.coords)
        stencil_values = map2unidimensional(value_up, value_down, independent_axis, stencil)
        init_coefficients = [np.mean(np.diff(stencil_values, degree)) for degree in range(self.degree + 1)]
        # init_coefficients[0] += np.min(stencil.coords, axis=0)[1 - independent_axis]
        yield CurvePolynomial(
            polynomial=Polynomial(init_coefficients),
            value_up=value_up,
            value_down=value_down,
            x_shift=coords[independent_axis] + 0.5
        )
