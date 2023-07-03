from typing import Dict, Tuple, Generator, Callable

import numpy as np
from numpy.polynomial import Polynomial

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd, CellCoords
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase, map2unidimensional
from lib.Curves.CurveBase import CurveBase
from lib.Curves.CurveCircle import CircleParams, CurveSemiCircle, get_concavity
from lib.Curves.CurvePolynomial import CurvePolynomial
from lib.StencilCreators import Stencil


class DefaultCircleCurveCellCreator(CurveCellCreatorBase):
    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[CurveSemiCircle, None, None]:
        value_up = regular_opposite_cells[1].evaluate(coords.coords)
        value_down = regular_opposite_cells[0].evaluate(coords.coords)
        x_points, stencil_values = map2unidimensional(value_up, value_down, independent_axis, stencil)
        concavity = get_concavity(x_points, stencil_values)
        yield CurveSemiCircle(
            CircleParams(
                x0=coords.tuple[independent_axis] + 0.5,
                y0=coords.tuple[1 - independent_axis] + 0.5 + concavity,
                radius=len(x_points)),
            value_up=value_up,
            value_down=value_down,
            concave=concavity > 0
        )


class DefaultPolynomialCurveCellCreator(CurveCellCreatorBase):

    def __init__(self, degree, regular_opposite_cell_searcher: Callable):
        super().__init__(regular_opposite_cell_searcher)
        self.degree = degree

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[CurveBase, None, None]:
        value_up = regular_opposite_cells[1].evaluate(coords.coords)
        value_down = regular_opposite_cells[0].evaluate(coords.coords)
        # x_points, stencil_values = map2unidimensional(value_up, value_down, independent_axis, stencil)
        init_coefficients = np.zeros(self.degree + 1)
        init_coefficients[0] = coords[1 - independent_axis] + 0.5
        # init_coefficients[0] = np.mean(stencil_values)
        # init_coefficients = [np.mean(np.diff(stencil_values, degree)) for degree in range(self.degree + 1)]
        # init_coefficients[0] += np.min(stencil.coords, axis=0)[1 - independent_axis]
        yield CurvePolynomial(
            polynomial=Polynomial(init_coefficients),
            value_up=value_up,
            value_down=value_down,
            x_shift=coords[independent_axis] + 0.5
        )
