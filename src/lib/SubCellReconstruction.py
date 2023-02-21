import itertools
from collections import namedtuple
from typing import Tuple, Union, List

import numpy as np

from lib.AuxiliaryStructures.GraphAuxiliaryFunctions import mesh_iterator
from lib.CellCreators.CellCreatorBase import CellBase
from lib.StencilCreators import StencilCreator
from src.Indexers import ArrayIndexerNd

CellCreatorPipeline = namedtuple("CellCreatorPipeline", "cell_iterator orientator stencil_creator cell_creator")


class ReconstructionErrorMeasureBase:
    def calculate_error(self, proposed_cell: CellBase, average_values: np.ndarray, indexer: ArrayIndexerNd,
                        smoothness_index: np.ndarray, independent_axis=0):
        return 0


class ReconstructionErrorMeasure(ReconstructionErrorMeasureBase):
    def __init__(self, stencil_creator: StencilCreator, metric: str = "l2", central_cell_extra_weight=0):
        self.stencil_creator = stencil_creator
        self.metric = metric
        self.central_cell_extra_weight = central_cell_extra_weight

    def calculate_error(self, proposed_cell: CellBase, average_values: np.ndarray, indexer: ArrayIndexerNd,
                        smoothness_index: np.ndarray, independent_axis=0):
        stencil = self.stencil_creator.get_stencil(average_values, smoothness_index, proposed_cell.coords,
                                                   independent_axis, indexer)
        kernel_vector_error = np.array([proposed_cell.integrate_rectangle(
            rectangle=np.array([coords, coords + 1])) - average_values[tuple(coords)] for coords in
                                        np.transpose(indexer[stencil.coords])])

        index_central_cell = np.where(np.all(proposed_cell.coords.array == stencil.coords, axis=1))[0][0]
        if self.metric in ["l2"]:
            loss = np.sum(kernel_vector_error ** 2) + \
                   self.central_cell_extra_weight * kernel_vector_error[index_central_cell] ** 2
        else:
            raise Exception(f"metric {self.metric} not implemented.")
        return loss


class SubCellReconstruction:
    def __init__(self, name, smoothness_calculator, reconstruction_error_measure,
                 cell_creators: List[CellCreatorPipeline], refinement: int = 1):
        self.name = name
        self.smoothness_calculator = smoothness_calculator
        self.reconstruction_error_measure = reconstruction_error_measure
        self.refinement = refinement
        self.cell_creators = cell_creators
        self.cells = dict()
        self.resolution = None

    def __str__(self):
        return self.name

    def fit(self, average_values: np.ndarray, indexer: ArrayIndexerNd):
        for r in range(self.refinement):
            self.resolution = np.shape(average_values)
            smoothness_index = self.smoothness_calculator(average_values, indexer)
            reconstruction_error = np.inf * np.ones(np.shape(smoothness_index))  # everything to be improved
            for cell_creator in self.cell_creators:
                new_cells = dict()
                for coords in cell_creator.cell_iterator(smoothness_index, reconstruction_error):
                    independent_axis = cell_creator.orientator.get_independent_axis(coords, average_values, indexer)
                    stencil = cell_creator.stencil_creator.get_stencil(
                        average_values, smoothness_index, coords, independent_axis, indexer)
                    for proposed_cell in cell_creator.cell_creator.create_cells(
                            average_values=average_values, indexer=indexer, cells=self.cells, coords=coords,
                            smoothness_index=smoothness_index, independent_axis=independent_axis, stencil=stencil):
                        proposed_cell_reconstruction_error = self.reconstruction_error_measure.calculate_error(
                            proposed_cell, average_values, indexer, smoothness_index, independent_axis)
                        if proposed_cell_reconstruction_error < reconstruction_error[coords.tuple]:
                            reconstruction_error[coords.tuple] = proposed_cell_reconstruction_error
                            new_cells[coords.tuple] = proposed_cell
                self.cells.update(new_cells)

            if r < self.refinement - 1:
                average_values = self.reconstruct_by_factor(resolution_factor=2)
        return self

    def reconstruct_by_factor(self, resolution_factor: Union[int, Tuple, np.ndarray] = 1):
        resolution_factor = np.array([resolution_factor] * len(self.resolution), dtype=int) \
            if isinstance(resolution_factor, int) else np.array(resolution_factor)
        average_values = np.zeros(resolution_factor * np.array(self.resolution, dtype=int))
        for ix in mesh_iterator(self.resolution, out_type=np.array):
            for sub_ix in mesh_iterator(resolution_factor, out_type=np.array):
                rectangle_upper_left_vertex = ix + sub_ix / resolution_factor
                rectangle_down_right_vertex = rectangle_upper_left_vertex + 1.0 / resolution_factor
                avg = self.cells[tuple(ix)].integrate_rectangle(
                    np.array([rectangle_upper_left_vertex, rectangle_down_right_vertex]))
                average_values[tuple(ix * resolution_factor + sub_ix)] = avg

        return average_values * np.prod(resolution_factor)

    def reconstruct_arbitrary_size(self, size: Union[Tuple, np.ndarray]):
        # TODO: test
        size = np.array(size)
        values = np.zeros(size)
        for ix in mesh_iterator(size, out_type=np.array):
            values[tuple(ix)] = self.cells[tuple(ix // self.resolution)](*(ix / self.resolution))
        return values
