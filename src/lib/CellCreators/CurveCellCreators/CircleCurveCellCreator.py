from typing import Dict, Generator, Tuple, Callable
import numpy as np
from numpy.polynomial import Polynomial

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase, map2unidimensional, get_zone
from lib.Curves.CurveBase import CurveBase
from lib.Curves.CurveCircle import CurveCirclePoints
from lib.Curves.CurvePolynomial import CurvePolynomial
from lib.StencilCreators import Stencil
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd


class CircleCurveCellCreator(CurveCellCreatorBase):
    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[CurveBase, None, None]:
        value_up = regular_opposite_cells[1].evaluate(coords.coords)
        value_down = regular_opposite_cells[0].evaluate(coords.coords)
        stencil_values = map2unidimensional(value_up, value_down, independent_axis, stencil)
        circle = CurveCirclePoints(
            *stencil_values,
            value_up=value_up,
            value_down=value_down,
            x_shift=coords[independent_axis] + 0.5
        )
        yield circle
