from typing import Dict, Tuple, Generator, Callable

import numpy as np
from numpy.polynomial import Polynomial

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd, CellCoords
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase, \
    prepare_stencil4one_dimensionalization, get_x_points, get_values_up_down_regcell_eval
from lib.Curves.CurveCircle import CircleParams, CurveSemiCircle, get_concavity
from lib.Curves.CurvePolynomial import CurvePolynomial
from lib.Curves.Curves import Curve
from lib.StencilCreators import Stencil


class DefaultCircleCurveCellCreator(CurveCellCreatorBase):
    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[CurveSemiCircle, None, None]:
        value_up, value_down = self.updown_value_getter(coords, regular_opposite_cells)
        stencil_values = prepare_stencil4one_dimensionalization(
            value_up=value_up, value_down=value_down, independent_axis=independent_axis, stencil=stencil,
            smoothness_index=smoothness_index, indexer=indexer)
        stencil_values = stencil_values.sum(axis=1)
        x_points = get_x_points(stencil, independent_axis)
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

    def __init__(self, degree, regular_opposite_cell_searcher: Callable,
                 updown_value_getter: Callable = get_values_up_down_regcell_eval):
        super().__init__(regular_opposite_cell_searcher, updown_value_getter=updown_value_getter)
        self.degree = degree

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up, value_down = self.updown_value_getter(coords, regular_opposite_cells)
        init_coefficients = np.zeros(self.degree + 1)
        init_coefficients[0] = coords[1 - independent_axis] + 0.5
        yield CurvePolynomial(
            polynomial=Polynomial(init_coefficients),
            value_up=value_up,
            value_down=value_down,
            x_shift=coords[independent_axis] + 0.5
        )
