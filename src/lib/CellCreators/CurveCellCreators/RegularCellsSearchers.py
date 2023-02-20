from typing import Dict, Tuple

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import CellBase
from lib.StencilCreators import get_regular_opposite_cell_coords
from src.Indexers import ArrayIndexerNd


def get_regular_opposite_cell_coords_sorted(coords: CellCoords, average_values: np.ndarray, dependent_axis: int,
                                            regularity_mask: np.ndarray, indexer: ArrayIndexerNd) \
        -> (Tuple[Tuple[int, int], Tuple[int, int]], np.ndarray):
    regular_opposite_cells, singular_cells = get_regular_opposite_cell_coords(coords, dependent_axis, regularity_mask,
                                                                              indexer)
    regular_opposite_cells = np.transpose(indexer[regular_opposite_cells])
    stencil_order = np.argsort([average_values[tuple(coord)] for coord in regular_opposite_cells])
    return regular_opposite_cells[stencil_order], singular_cells


def get_regular_opposite_cells(coords: CellCoords, dependent_axis: int, average_values: np.ndarray,
                               smoothness_index, indexer: ArrayIndexerNd,
                               cells: Dict[Tuple[int], CellBase], threshold=0.5):
    regular_opposite_cell_coords, _ = get_regular_opposite_cell_coords_sorted(coords, average_values, dependent_axis,
                                                                              smoothness_index >= threshold, indexer)
    regular_opposite_cells = tuple([cells[tuple(o)] for o in regular_opposite_cell_coords])
    return tuple(sorted(regular_opposite_cells, key=lambda roc: roc.coords[dependent_axis]))
