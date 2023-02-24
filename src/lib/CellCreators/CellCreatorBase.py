from collections import OrderedDict
from typing import Dict, Tuple, List, Generator

import numpy as np
from lib.AuxiliaryStructures.Constants import NEIGHBOURHOOD_8, neighbourhood_8_ix, NIGHT_NAVY, \
    ELEGANT_EGGPLANT, OLD_OLIVE, REAL_RED
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from src.Indexers import ArrayIndexerNd

from src.lib.StencilCreators import Stencil

REGULAR_CELL_TYPE = "RegularCell"
CURVE_CELL_TYPE = "CurveCell"
VERTEX_CELL_TYPE = "VertexCell"
TRIPLE_POINT_CELL_TYPE = "TriplePointCell"

SPECIAL_CELLS_COLOR_DICT = OrderedDict([
    (REGULAR_CELL_TYPE, NIGHT_NAVY),
    (CURVE_CELL_TYPE, REAL_RED),
    (VERTEX_CELL_TYPE, OLD_OLIVE),
    (TRIPLE_POINT_CELL_TYPE, ELEGANT_EGGPLANT)
])


# ======================================== #
#           Cell Base definition
# ======================================== #
class CellBase:
    CELL_TYPE = None

    def __init__(self, coords: CellCoords):
        self.coords = coords

    def __str__(self):
        return type(self).__name__

    @property
    def center_of_cell(self):
        return self.coords + 0.5

    def flux(self, velocity: np.ndarray, indexer: ArrayIndexerNd) -> Dict[Tuple, float]:
        assert np.all(np.abs(velocity) <= 1), "velocity should be less than 1 on each coordinate."
        return {tuple(indexer[next_coord]): self.integrate_rectangle(rectangle) for next_coord, rectangle in
                zip(*get_rectangles_and_coords_to_calculate_flux(np.array(self.coords.coords), velocity))}

    def integrate_rectangle(self, rectangle) -> float:
        raise Exception("Not implemented.")

    def evaluate(self, query_points: np.ndarray) -> np.ndarray:
        raise Exception("Not implemented.")


# ======================================== #
#        Cell creator definition
# ======================================== #
class CellCreatorBase:
    def create_cells(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[Tuple[int, ...], CellBase],
                     coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                     stencil: Stencil, stencils: Dict[Tuple[int, ...], np.ndarray]) -> Generator[CellBase, None, None]:
        raise Exception("Not implemented")

    def __str__(self):
        return self.__class__.__name__


# ======================================== #
#           Auxiliary functions
# ======================================== #
def velocity_8nei_direction(velocity):
    return np.sign(velocity)


def get_relative_next_coords_to_calculate_flux(velocity):
    return NEIGHBOURHOOD_8[(neighbourhood_8_ix(velocity_8nei_direction(velocity)) + np.array([-1, 0, 1])) % 8]


def get_relative_rectangle_to_calculate_flux(velocity, coords):
    middle_point = velocity_8nei_direction(velocity) - velocity
    coords1 = coords + 1 - np.sign(velocity)
    coords2 = [vertex_1 + np.sign(middle_point - vertex_1) * np.min(([1, 1], np.abs(middle_point - vertex_1)), axis=0)
               for vertex_1 in coords1]
    rectangles = np.sort([coords1, coords2], axis=0).swapaxes(0, 1)
    rectangles[rectangles < 0] = 0
    rectangles[rectangles > 1] = 1
    return rectangles


def get_rectangles_and_coords_to_calculate_flux(coords: np.ndarray, velocity: np.ndarray) -> \
        (Tuple[int], List[np.ndarray]):
    """
    The returned rectangles have the limiting points [(x0, y0), (xf, yf)].
    """
    next_coords = get_relative_next_coords_to_calculate_flux(velocity)
    next_rectangles = get_relative_rectangle_to_calculate_flux(velocity, next_coords)
    next_coords += coords
    next_rectangles += coords
    return next_coords, next_rectangles
