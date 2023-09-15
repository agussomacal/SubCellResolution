import copy
import itertools
import time
from collections import defaultdict, namedtuple
from contextlib import contextmanager
from dataclasses import dataclass
from logging import warning
from typing import Tuple, Union, List, Dict

import numpy as np
from scipy.optimize import minimize

from lib.AuxiliaryStructures.Constants import CURVE_CELL
from lib.AuxiliaryStructures.GraphAuxiliaryFunctions import mesh_iterator
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CellBase, REGULAR_CELL_TYPE
from lib.StencilCreators import StencilCreator, Stencil

CellCreatorPipeline = namedtuple("CellCreatorPipeline",
                                 "cell_iterator orientator stencil_creator cell_creator reconstruction_error_measure",
                                 defaults=(None,) * 5)


class ReconstructionErrorMeasureBase:
    def calculate_error(self, proposed_cell: CellBase, average_values: np.ndarray, indexer: ArrayIndexerNd,
                        smoothness_index: np.ndarray, independent_axis=0, stencil: Stencil = None):
        return 0


def keep_all_cells(stencil_coords, *args, **kwargs):
    return stencil_coords


def keep_cells_on_condition(stencil_coords, smoothness_index, indexer, condition):
    return [c for c in stencil_coords if condition(smoothness_index[indexer[c]])]


def curve_condition(smoothness):
    return smoothness == CURVE_CELL


class ReconstructionErrorMeasure(ReconstructionErrorMeasureBase):
    def __init__(self, stencil_creator: StencilCreator, metric: int = 2, central_cell_extra_weight=0,
                 keeping_cells_condition=keep_all_cells):
        self.stencil_creator = stencil_creator
        self.metric = metric
        self.central_cell_extra_weight = central_cell_extra_weight
        self.keeping_cells_condition = keeping_cells_condition

    def calculate_error(self, proposed_cell: CellBase, average_values: np.ndarray, indexer: ArrayIndexerNd,
                        smoothness_index: np.ndarray, independent_axis=0, stencil: Stencil = None):
        if stencil is None:
            stencil = self.stencil_creator.get_stencil(
                average_values=average_values, smoothness_index=smoothness_index, coords=proposed_cell.coords,
                independent_axis=independent_axis, indexer=indexer)
        stencil_coords = self.keeping_cells_condition(stencil.coords, smoothness_index=smoothness_index,
                                                      indexer=indexer)
        kernel_vector_error = np.array([
            proposed_cell.integrate_rectangle(rectangle=np.array([coords, coords + 1]))
            - average_values[indexer[coords]]
            for coords in stencil_coords])

        loss = np.sum(np.abs(kernel_vector_error) ** self.metric)

        cc = np.all(proposed_cell.coords.array == stencil_coords, axis=1)
        if np.any(cc):
            index_central_cell = np.where(cc)[0][0]
            loss += self.central_cell_extra_weight * np.abs(kernel_vector_error[index_central_cell]) ** self.metric
        return loss


class ReconstructionErrorMeasureDefaultStencil(ReconstructionErrorMeasure):
    def calculate_error(self, proposed_cell: CellBase, average_values: np.ndarray, indexer: ArrayIndexerNd,
                        smoothness_index: np.ndarray, independent_axis=0, stencil: Stencil = None):
        return super().calculate_error(proposed_cell, average_values, indexer, smoothness_index, independent_axis)


def ddf():
    return defaultdict(float)


class SubCellReconstruction:
    def __init__(self, name, smoothness_calculator, reconstruction_error_measure=ReconstructionErrorMeasureBase,
                 cell_creators: List[CellCreatorPipeline] = [], refinement: int = 1, obera_iterations=0):
        self.name = name
        self.smoothness_calculator = smoothness_calculator
        self.reconstruction_error_measure = reconstruction_error_measure
        self.refinement = refinement
        self.cell_creators = cell_creators
        self.cells = dict()
        self.stencils = dict()
        self.resolution = None
        self.obera_iterations = obera_iterations

        self.times = defaultdict(ddf)
        self.obera_fevals = defaultdict(ddf)

    def __str__(self):
        return self.name

    @contextmanager
    def cell_timer(self, coords, cell_creator):
        if self.refinement > 1: warning("Time calculations won't be correct if refinement > 1")
        t0 = time.time()
        yield
        self.times[str(cell_creator)][coords.tuple] = time.time() - t0

    def fit(self, average_values: np.ndarray, indexer: ArrayIndexerNd):
        for r in range(self.refinement):
            self.cells = dict()
            self.stencils = dict()
            self.resolution = np.shape(average_values)
            smoothness_index = self.smoothness_calculator(average_values, indexer)
            reconstruction_error = np.inf * np.ones(np.shape(smoothness_index))  # everything to be improved
            for i, cell_creator in enumerate(self.cell_creators):
                for coords in cell_creator.cell_iterator(smoothness_index=smoothness_index,
                                                         reconstruction_error=reconstruction_error):
                    if self.refinement > 1: warning("Time calculations won't be correct if refinement > 1")
                    t0 = time.time()
                    # with self.cell_timer(coords, cell_creator):

                    independent_axis = cell_creator.orientator.get_independent_axis(coords, average_values, indexer)
                    stencil = cell_creator.stencil_creator.get_stencil(
                        average_values, smoothness_index, coords, independent_axis, indexer)
                    proposed_cells = list(cell_creator.cell_creator.create_cells(
                        average_values=average_values, indexer=indexer, cells=self.cells, coords=coords,
                        smoothness_index=smoothness_index, independent_axis=independent_axis, stencil=stencil,
                        stencils=self.stencils))
                    # only calculate error if more than one proposition is done otherwise just keep the only one
                    if i > 0 or len(proposed_cells) > 1:
                        for proposed_cell in proposed_cells:
                            if isinstance(proposed_cell, tuple):
                                proposed_cell, coords = proposed_cell

                            # ---------- Doing OBERA ---------- #
                            if proposed_cell.CELL_TYPE != REGULAR_CELL_TYPE and self.obera_iterations > 0:
                                def optim_func(params):
                                    proposed_cell.curve.params = params
                                    loss = self.reconstruction_error_measure.calculate_error(
                                        proposed_cell, average_values, indexer, smoothness_index, independent_axis)
                                    return loss

                                # number of function evaluation without gradient is twice the number of parameters
                                x0 = np.ravel(proposed_cell.curve.params)
                                res = minimize(optim_func, x0=x0, method="L-BFGS-B", tol=1e-10,
                                               options={'maxiter': self.obera_iterations * 2 * (1 + len(x0))})
                                proposed_cell.curve.params = res.x
                                self.obera_fevals[proposed_cell.CELL_TYPE][coords.tuple] += res.nfev

                            # ---------- Deciding which cell to keep ---------- #
                            # if some other cell has been put there than compare
                            if coords.tuple in self.stencils:
                                if cell_creator.reconstruction_error_measure is None:
                                    reconstruction_error_measure = copy.copy(self.reconstruction_error_measure)
                                else:
                                    reconstruction_error_measure = copy.copy(cell_creator.reconstruction_error_measure)
                                proposed_cell_reconstruction_error = reconstruction_error_measure.calculate_error(
                                    proposed_cell, average_values, indexer, smoothness_index, independent_axis,
                                    stencil)

                                # if it has never been calculated or the stencil used is different from the current
                                if np.isinf(reconstruction_error[coords.tuple]) or set(
                                        list(map(tuple, stencil.coords.tolist()))) != set(self.stencils[coords.tuple]):
                                    old_cell_reconstruction_error = reconstruction_error_measure.calculate_error(
                                        self.cells[coords.tuple], average_values, indexer, smoothness_index,
                                        independent_axis, stencil)
                                else:
                                    old_cell_reconstruction_error = reconstruction_error[coords.tuple]

                                if proposed_cell_reconstruction_error < old_cell_reconstruction_error:
                                    reconstruction_error[coords.tuple] = proposed_cell_reconstruction_error
                                    self.cells[coords.tuple] = proposed_cell
                                    self.stencils[coords.tuple] = list(map(tuple, stencil.coords.tolist()))
                            else:
                                self.cells[coords.tuple] = proposed_cell
                                self.stencils[coords.tuple] = list(map(tuple, stencil.coords.tolist()))

                    else:
                        proposed_cell = proposed_cells.pop()
                        self.cells[coords.tuple] = proposed_cell
                    self.times[proposed_cell.CELL_TYPE][coords.tuple] += time.time() - t0

            if r < self.refinement - 1:
                average_values = self.reconstruct_by_factor(resolution_factor=2)
                indexer = ArrayIndexerNd(average_values, indexer.modes)
        return self

    def reconstruct_by_factor(self, resolution_factor: Union[int, Tuple, np.ndarray] = 1):
        """
        Uses averages to reconstruct.
        :param resolution_factor:
        :return:
        """
        return reconstruct_by_factor(cells=self.cells, resolution=self.resolution, resolution_factor=resolution_factor)

    def reconstruct_arbitrary_size(self, size: Union[Tuple, np.ndarray]):
        """
        Uses evaluation to reconstruct.
        :param size:
        :return:
        """
        return reconstruct_arbitrary_size(cells=self.cells, resolution=self.resolution, size=size)


def reconstruct_arbitrary_size(cells: Dict[Tuple[int, ...], CellBase], resolution, size: Union[Tuple, np.ndarray],
                               cells2reconstruct: List[Tuple] = None):
    """
    Uses evaluation to reconstruct.
    :param size:
    :return:
    """
    size = np.array(size)
    values = np.zeros(size)
    for ix in itertools.product(*list(map(range, size))):
        cell_ix = tuple(map(int, np.array(ix) / size * resolution))
        if cells2reconstruct is None or cell_ix in cells2reconstruct:
            values[ix] = cells[cell_ix].evaluate(
                (np.array(ix) / size * resolution)[np.newaxis, :])
    return values


def reconstruct_by_factor(cells: Dict[Tuple[int, ...], CellBase], resolution,
                          resolution_factor: Union[int, Tuple, np.ndarray] = 1,
                          cells2reconstruct: List[Tuple] = None):
    """
    Uses averages to reconstruct.
    :param resolution_factor:
    :return:
    """
    resolution_factor = np.array([resolution_factor] * len(resolution), dtype=int) \
        if isinstance(resolution_factor, int) else np.array(resolution_factor)
    average_values = np.zeros(resolution_factor * np.array(resolution, dtype=int))
    for ix in mesh_iterator(resolution, out_type=np.array) if cells2reconstruct is None else cells2reconstruct:
        for sub_ix in mesh_iterator(resolution_factor, out_type=np.array):
            rectangle_upper_left_vertex = ix + sub_ix / resolution_factor
            rectangle_down_right_vertex = rectangle_upper_left_vertex + 1.0 / resolution_factor
            avg = cells[tuple(ix)].integrate_rectangle(
                np.array([rectangle_upper_left_vertex, rectangle_down_right_vertex]))
            average_values[tuple(ix * resolution_factor + sub_ix)] = avg
    return average_values * np.prod(resolution_factor)
