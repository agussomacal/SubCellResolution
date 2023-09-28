from typing import Generator, Tuple, Dict, Callable

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords, ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CellBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import CurveCellCreatorBase, \
    prepare_stencil4one_dimensionalization, get_values_up_down_regcell_eval
from lib.Curves.Curves import Curve
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import CURVE_PROBLEM
from lib.DataManagers.LearningMethodManager import LearningMethodManager
from lib.StencilCreators import Stencil


class LearningCurveCellCreator(CurveCellCreatorBase):
    def __init__(self, learning_manager: LearningMethodManager, regular_opposite_cell_searcher: Callable,
                 updown_value_getter: Callable = get_values_up_down_regcell_eval):
        super().__init__(regular_opposite_cell_searcher=regular_opposite_cell_searcher,
                         updown_value_getter=updown_value_getter)
        assert isinstance(learning_manager, LearningMethodManager), \
            "learning method should be a FluxLearning method"
        assert learning_manager.type_of_problem == CURVE_PROBLEM, "Should be {}".format(CURVE_PROBLEM)
        self.learning_manager = learning_manager

    def create_curves(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[str, CellBase],
                      coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                      stencil: Stencil, regular_opposite_cells: Tuple) -> Generator[Curve, None, None]:
        value_up, value_down = self.updown_value_getter(coords, regular_opposite_cells)
        stencil_values = prepare_stencil4one_dimensionalization(independent_axis, value_up, value_down, stencil,
                                                                smoothness_index, indexer)
        curve_params = self.learning_manager.predict_curve_params(kernel=stencil_values)
        curve = self.learning_manager.dataset_manager.create_curve_from_params(
            curve_params=curve_params,
            coords=coords,
            independent_axis=independent_axis,
            value_up=value_up,
            value_down=value_down,
            stencil=stencil
        )
        # needs to be in the middle because learning was done assuming (0, 0) is in the middle.
        curve.set_x_shift(np.mean(stencil.coords[:, independent_axis]) + 0.5)
        curve.set_y_shift(np.mean(stencil.coords[:, 1 - independent_axis]) + 0.5)
        yield curve
