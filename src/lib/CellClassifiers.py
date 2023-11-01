from typing import List

from lib.AuxiliaryStructures.Constants import CURVE_CELL
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords, ArrayIndexerNd
from lib.DataManagers.LearningMethodManager import LearningMethodManager
from lib.StencilCreators import get_fixed_stencil_values


def cell_classifier_try_it_all(coords: CellCoords, average_values, smoothness_index, indexer: ArrayIndexerNd,
                               cell_creators: List, **kwargs):
    return list(range(len(cell_creators)))


def cell_classifier_by_smoothness(coords: CellCoords, average_values, smoothness_index, indexer: ArrayIndexerNd,
                                  cell_creators: List, regular_cell_creators_indexes=[0],
                                  singular_cell_creators_indexes=None,
                                  **kwargs):
    """
    If regular cell then try only the regular cell creator otherwise try all the curve types.
    :param coords:
    :param average_values:
    :param smoothness_index:
    :param cell_creators:
    :param kwargs:
    :return:
    """
    return ([i for i in range(len(cell_creators)) if i not in regular_cell_creators_indexes]
            if singular_cell_creators_indexes is None else singular_cell_creators_indexes) \
        if smoothness_index[coords.tuple] == CURVE_CELL else regular_cell_creators_indexes


def cell_classifier_ml(coords: CellCoords, average_values, smoothness_index, indexer: ArrayIndexerNd,
                       cell_creators: List, ml_model: LearningMethodManager,
                       regular_cell_creators_indexes=[0], damping=None, **kwargs):
    """
    If regular cell then try only the regular cell creator otherwise try all the curve types.
    :param coords:
    :param average_values:
    :param smoothness_index:
    :param cell_creators:
    :param kwargs:
    :return:
    """

    kernel = get_fixed_stencil_values(stencil_size=ml_model.dataset_manager.kernel_size,
                                      coords=coords, average_values=average_values,
                                      indexer=indexer)
    # TODO: harcoded
    # To penalize if models tend to to predict vertex too much
    if smoothness_index[coords.tuple] == CURVE_CELL:
        index = ml_model.predict_curve_type_index(kernel, damping=damping)
        res = [index + max(regular_cell_creators_indexes) + 1]
        if (damping is not None) and (damping[index] != 1):
            # if predicting vertex add also in case other curve for if it fails.
            res.append(index - 1)
        return res
    else:
        return regular_cell_creators_indexes
