from collections import namedtuple
from typing import Tuple, Dict, Generator, Callable

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellCreatorBase, \
    CellBase, CURVE_CELL_TYPE
from lib.Curves.CurveBase import CurveBase
from lib.StencilCreators import Stencil
from src.Indexers import ArrayIndexerNd


# ====================================================== #
#                    Curve cell Base                     #
# ====================================================== #
class CellCurveBase(CellBase):
    CELL_TYPE = CURVE_CELL_TYPE

    def __init__(self, coords: CellCoords, curve: CurveBase, regular_opposite_cells: Tuple[CellBase, CellBase],
                 dependent_axis: int = 1):
        super().__init__(coords)

        assert len(self.coords.coords) < 3, "This method only works in 2D"
        self.curve = curve
        self.dependent_axis = dependent_axis
        self.independent_axis = 1 - dependent_axis

        # evaluates where the regular cells are positioned with respect of the curve, if above or below.
        self.regular_opposite_cells = regular_opposite_cells

    def predict_regular_index(self, query_point):
        # is it the regular cell of above or of below.
        return int(self.curve(query_point[self.independent_axis], query_point[self.dependent_axis]))

    def integrate_rectangle(self, rectangle) -> float:
        return self.curve.calculate_rectangle_average(
            x_limits=rectangle[:, self.independent_axis],
            y_limits=rectangle[:, self.dependent_axis]
        )

    def evaluate(self, query_points: np.ndarray) -> np.ndarray:
        return np.array(
            [self.regular_opposite_cells[self.predict_regular_index(point)].evaluate(point) for point in query_points])


# ====================================================== #
#                  Curve cell creator base               #
# ====================================================== #
CurveAlternative = namedtuple("CurveAlternative", ["cell", "stencil"])


class CurveCellCreatorBase(CellCreatorBase):
    def __init__(self, regular_opposite_cell_searcher: Callable):
        self.regular_opposite_cell_searcher = regular_opposite_cell_searcher

    def create_cells(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                     coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                     stencil: Stencil) -> Generator[CellBase, None, None]:
        regular_opposite_cells = self.regular_opposite_cell_searcher(
            coords=coords, dependent_axis=1 - independent_axis, average_values=average_values,
            smoothness_index=smoothness_index, indexer=indexer, cells=cells
        )
        for curve in self.create_curves(average_values, indexer, cells, coords, smoothness_index,
                                        independent_axis, stencil, regular_opposite_cells):
            yield CellCurveBase(
                coords=coords,
                curve=curve,
                regular_opposite_cells=regular_opposite_cells,
                dependent_axis=1 - independent_axis)

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[CurveBase, None, None]:
        raise Exception("Not implemented.")
