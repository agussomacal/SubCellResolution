import copy
import itertools
from functools import partial
from typing import Union, Tuple, Dict

import numpy as np
from tqdm import tqdm

from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import get_rectangles_and_coords_to_calculate_flux
from lib.CellCreators.RegularCellCreator import CellRegularBase
from lib.SmoothnessCalculators import oracle
from lib.SubCellReconstruction import SubCellReconstruction


class SubCellScheme:
    def __init__(self, name, subcell_reconstructor: SubCellReconstruction, min_value=None, max_value=None):
        self.name = name
        self.subcell_reconstructor = subcell_reconstructor
        self.min_value = min_value
        self.max_value = max_value

    def __str__(self):
        return self.name

    @property
    def resolution(self):
        return self.subcell_reconstructor.resolution

    def evolve(self, init_average_values: np.ndarray, indexer: ArrayIndexerNd, velocity: np.ndarray, ntimes: int,
               interface_oracle: np.ndarray = None):
        """
        Assumes dt = 1 and velocity is relative to the size of a cell so 1 means that in 1 time step will do one cell.
        """
        solution = [init_average_values]
        all_cells = []
        for i in tqdm(range(ntimes), "Evolving {}...".format(self)):
            average_values = np.copy(solution[-1])
            if interface_oracle is not None:
                self.subcell_reconstructor.smoothness_calculator = partial(oracle, mask=interface_oracle[i])
            self.subcell_reconstructor.fit(average_values, indexer)
            all_cells.append(self.subcell_reconstructor.cells.copy())

            # ----- update values with calculated fluxes ----- #
            for coords_i, cell in self.subcell_reconstructor.cells.items():
                v = velocity if len(np.shape(velocity)) == 1 else velocity[coords_i]
                for coords_j, flux in cell.flux(v, indexer).items():
                    if flux != 0:
                        # if self.min_value is not None and average_values[coords_i] - self.min_value < flux:
                        #     flux = average_values[coords_i] - self.min_value
                        # if self.max_value is not None and self.max_value - average_values[coords_j] < flux:
                        #     flux = self.max_value - average_values[coords_j]
                        average_values[coords_i] -= flux
                        average_values[coords_j] += flux
            if self.min_value is not None:
                average_values[average_values < self.min_value] = self.min_value
            if self.max_value is not None:
                average_values[average_values > self.max_value] = self.max_value
            solution.append(average_values)

        return solution, all_cells


class CellUpWind(CellRegularBase):
    def __init__(self, coords: CellCoords, flux: float):
        super().__init__(coords)
        # TODO: wrong for directions
        self.precalculated_flux = flux

    def flux(self, velocity: np.ndarray, indexer: ArrayIndexerNd) -> Dict[Tuple, float]:
        next_coords, _ = get_rectangles_and_coords_to_calculate_flux(np.array(self.coords.coords), velocity)
        return {tuple(indexer[nxt_coords]): self.precalculated_flux for nxt_coords in next_coords}


class UpWindScheme:
    def __init__(self):
        self.name = "UpWind"

    def __str__(self):
        return self.name

    def evolve(self, init_average_values: np.ndarray, indexer: ArrayIndexerNd, velocity: np.ndarray, ntimes: int):
        # TODO: not repeat code, use the basic structure of Scheme methods
        assert np.prod(np.sign(velocity)) == 0, "Actual code only supports velocity in one direction only"
        solution = [init_average_values]
        all_cells = []
        for _ in tqdm(range(ntimes), "Evolving {}...".format(self)):
            average_values = np.copy(solution[-1])

            f = 0 * average_values
            for axis, v in enumerate(velocity):
                fluxes = average_values * v
                f += fluxes  # only works if v is in x or y direction.
                average_values -= fluxes
                average_values += np.roll(fluxes, shift=int(np.sign(v)), axis=axis)

            all_cells.append({coords: CellUpWind(CellCoords(coords), f[coords]) for coords in
                              itertools.product(*list(map(range, np.shape(init_average_values))))})
            solution.append(average_values)

        return solution, all_cells
