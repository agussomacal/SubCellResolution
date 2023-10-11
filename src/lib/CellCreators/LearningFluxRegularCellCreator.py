from typing import Dict, Tuple, Generator, Callable

import numpy as np

from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords, ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CellBase, CellCreatorBase
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import get_values_up_down_regcell_eval, \
    get_values_up_down_maxmin, prepare_stencil4one_dimensionalization
from lib.CellCreators.RegularCellCreator import CellRegularBase
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import FLUX_PROBLEM
from lib.DataManagers.LearningMethodManager import LearningMethodManager
from lib.StencilCreators import Stencil


# class CellLearnedFlux(CellRegularBase):
#     def __init__(self, coords: CellCoords, next_coords, next_fluxes):
#         super().__init__(coords)
#         self.next_coords = next_coords
#         self.next_fluxes = next_fluxes
#
#     def flux(self, velocity: np.ndarray, indexer: ArrayIndexerNd) -> Dict[Tuple, float]:
#         return {tuple(indexer[nxt_coords]): next_flux for nxt_coords, next_flux in
#                 zip(self.next_coords, self.next_fluxes)}
#
#     def evaluate(self, query_points: np.ndarray):
#         raise Exception("Not implemented.")


class CellLearnedFlux(CellRegularBase):
    def __init__(self, coords, flux_calculator: Callable):
        super().__init__(coords)
        self.flux_calculator = flux_calculator

    def flux(self, velocity: np.ndarray, indexer: ArrayIndexerNd) -> Dict[Tuple, float]:
        next_coords, next_fluxes = self.flux_calculator(velocity)
        # TODO: Harcoded velocities
        return {tuple(indexer[next_coords[0]]): next_fluxes}
        # return {tuple(indexer[nxt_coords]): next_flux for nxt_coords, next_flux in
        #         zip(next_coords, next_fluxes)}

    def evaluate(self, query_points: np.ndarray):
        raise Exception("Not implemented.")


class LearningFluxRegularCellCreator(CellCreatorBase):
    def __init__(self, learning_manager: LearningMethodManager,
                 updown_value_getter: Callable):
        assert isinstance(learning_manager, LearningMethodManager), \
            "learning method should be a FluxLearning method"
        assert learning_manager.type_of_problem == FLUX_PROBLEM
        self.learning_manager = learning_manager
        self.updown_value_getter = updown_value_getter

    def create_cells(self, average_values: np.ndarray, indexer: ArrayIndexerNd, cells: Dict[Tuple[int, ...], CellBase],
                     coords: CellCoords, smoothness_index: np.ndarray, independent_axis: int,
                     stencil: Stencil, stencils: Dict[Tuple[int, ...], np.ndarray]) -> Generator[CellBase, None, None]:
        value_up, value_down = self.updown_value_getter(coords, stencil=stencil)
        stencil_values = prepare_stencil4one_dimensionalization(independent_axis, value_up, value_down, stencil,
                                                                smoothness_index, indexer)

        def flux_calculator(velocity):
            return self.learning_manager.predict_flux(stencil_values, velocity)

        yield CellLearnedFlux(coords, flux_calculator=flux_calculator)
