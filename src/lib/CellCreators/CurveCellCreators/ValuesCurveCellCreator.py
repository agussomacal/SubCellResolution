from typing import Dict, Generator, Tuple, Callable, Type

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase, map2unidimensional
from lib.Curves.CurveBase import CurveBase, CurveReparametrized
from lib.StencilCreators import Stencil


class ValuesCurveCellCreator(CurveCellCreatorBase):
    def __init__(self, vander_curve: Type[CurveReparametrized], regular_opposite_cell_searcher: Callable):
        super().__init__(regular_opposite_cell_searcher)
        self.vander_curve = vander_curve

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[CurveBase, None, None]:
        value_up = regular_opposite_cells[1].evaluate(coords.coords)
        value_down = regular_opposite_cells[0].evaluate(coords.coords)
        x_points, stencil_values = map2unidimensional(value_up, value_down, independent_axis, stencil)

        yield self.vander_curve(
            x_points=x_points,
            y_points=stencil_values,
            value_up=value_up,
            value_down=value_down,
        )
