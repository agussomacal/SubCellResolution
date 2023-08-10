from typing import Generator, List, Tuple, Dict, Callable

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords, ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase, CellCurveBase
from lib.CellOrientators import BaseOrientator
from lib.Curves.Curves import Curve
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import CURVE_PROBLEM
from lib.DataManagers.LearningMethodManager import LearningMethodManager
from lib.StencilCreators import StencilCreatorFixedShape, Stencil


class LearningCurveCellCreator(CurveCellCreatorBase):
    def __init__(self, learning_manager: LearningMethodManager, regular_opposite_cell_searcher: Callable):
        super().__init__(regular_opposite_cell_searcher=regular_opposite_cell_searcher)
        assert isinstance(learning_manager, LearningMethodManager), \
            "learning method should be a FluxLearning method"
        assert learning_manager.type_of_problem == CURVE_PROBLEM, "Should be {}".format(CURVE_PROBLEM)
        self.learning_manager = learning_manager

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:

        value_up = regular_opposite_cells[1].evaluate(coords.coords)
        value_down = regular_opposite_cells[0].evaluate(coords.coords)
        # x_points, stencil_values = map2unidimensional(value_up, value_down, independent_axis, stencil)
        # if the values are not 0 or 1
        min_value = np.min((value_up, value_down))
        stencil_values = stencil.values - np.min(stencil.values)
        stencil_values /= np.max(stencil_values)
        curve_params = self.learning_manager.predict_curve_params(kernel=stencil.values.reshape((3, 3)))
        curve = self.learning_manager.dataset_manager.create_curve_from_params(
            curve_params=curve_params,
            coords=coords,
            independent_axis=independent_axis,
            value_up=value_up,
            value_down=value_down
        )

        yield curve

