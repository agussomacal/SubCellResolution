import operator
import time
from functools import partial

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot, one_line_iterator, perplex_plot
from experiments.VizReconstructionUtils import plot_cells, draw_cell_borders, plot_cells_identity, \
    plot_cells_vh_classification_core, plot_cells_not_regular_classification_core, plot_curve_core, plot_specific_cells, \
    SpecialCellsPlotTuple, transform_points2plot
from experiments.subcell_paper.global_params import CurveAverageQuadraticCC, CCExtraWeight, cpink, corange, cyellow, \
    cblue, cgreen, runsinfo, cbrown, cgray, cpurple, cred, ccyan, EVALUATIONS, angle_threshold
from experiments.subcell_paper.tools import get_reconstruction_error, calculate_averages_from_image, load_image, \
    reconstruct, singular_cells_mask
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import REGULAR_CELL_TYPE
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells, \
    get_opposite_regular_cells_by_stencil, get_regular_opposite_cell_coords_by_minmax
from lib.CellCreators.CurveCellCreators.TaylorCurveCellCreator import TaylorCircleCurveCellCreator
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator, \
    ValuesLineConsistentCurveCellCreator
from lib.CellCreators.CurveCellCreators.VertexCellCreator import LinearVertexCellCurveCellCreator
from lib.CellCreators.RegularCellCreator import PiecewiseConstantRegularCellCreator, MirrorCellCreator
from lib.CellCreators.VertexCellCreators.VertexCellCreatorBase import VertexCellCreatorUsingNeighboursLines
from lib.CellIterators import iterate_by_reconstruction_error_and_smoothness, \
    iterate_all
from lib.CellOrientators import BaseOrientator, OrientByGradient, OrientPredefined
from lib.SmoothnessCalculators import naive_piece_wise
from lib.StencilCreators import StencilCreatorAdaptive, StencilCreatorFixedShape
from lib.SubCellReconstruction import SubCellReconstruction, ReconstructionErrorMeasure, CellCreatorPipeline, \
    keep_cells_on_condition, curve_condition, ReconstructionErrorMeasureDefaultStencil
from lib.SubCellScheme import SubCellScheme

# ========== ========== Names and colors to present ========== ========== #
names_dict = {
    "x": "X independent",
    "y": "Y independent",
    "mixed": "Mixed",
}
orientation_colors = [cgreen, cblue, cred]
orient_names = ["X independent", "Y independent", "Mixed"]


# ========== ========== Experiment definitions ========== ========== #
def experiment(image, num_cells_per_dim):
    image_array = load_image(image)
    avg_values = calculate_averages_from_image(image_array, num_cells_per_dim)
    smoothness_index = singular_cells_mask(avg_values)
    indexer = ArrayIndexerNd(avg_values, "cyclic")
    regular_cells = dict()
    singular_cells = dict()
    for coords in iterate_by_reconstruction_error_and_smoothness(
            reconstruction_error=np.zeros(np.shape(avg_values)),
            smoothness_index=smoothness_index,
            value=CURVE_CELL, condition=operator.eq):
        regular_cells[coords.tuple], singular_cells[coords.tuple] = get_regular_opposite_cell_coords_by_minmax(
            coords, avg_values, indexer,
            acceptance_criterion=lambda c: smoothness_index[indexer[c]] == 0)

    return {
        "opposite_cells": regular_cells,
    }


@perplex_plot()
@one_line_iterator
def plot_opposite_cells(fig, ax, image, num_cells_per_dim, opposite_cells,
                        alpha=0.5, cmap="Greys_r", c=cblue, marker=".",
                        linestyle="--", linewidth=3, trim=((0, 0), (0, 0)), numbers_on=True):
    image = load_image(image)
    mesh_shape = (num_cells_per_dim, num_cells_per_dim)
    plot_cells(ax, colors=image, mesh_shape=mesh_shape, alpha=alpha, cmap=cmap,
               vmin=np.min(image), vmax=np.max(image))

    avg_values = calculate_averages_from_image(image, num_cells_per_dim)
    plot_specific_cells(
        ax=ax,
        mesh_shape=mesh_shape,
        special_cells=[
            SpecialCellsPlotTuple(name="CurveCells", indexes=list(zip(*np.where(singular_cells_mask(avg_values)))),
                                  color=cgreen, alpha=0.5)

        ],
        rectangle_mode=False)
    for ccenter, oppcells in opposite_cells.items():
        ax.plot(*np.transpose(transform_points2plot([oppcells[0], ccenter, oppcells[1]]) + 0.5), c=c,
                linestyle=linestyle, linewidth=linewidth, marker=marker)

    draw_cell_borders(
        ax, mesh_shape=num_cells_per_dim,
        refinement=np.array(mesh_shape) // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )
    ax.set_xlim((-0.5 + trim[0][0], mesh_shape[0] - trim[0][1] - 0.5))
    ax.set_ylim((mesh_shape[1] - trim[1][0] - 0.5, trim[1][1] - 0.5))


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='RegularOppositeCells',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()

    lab.define_new_block_of_functions(
        "experiment_oppregcells",
        experiment,
        recalculate=False
    )

    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=15,
        num_cells_per_dim=[10, 15, 20, 28, 30, 42],  # 60
        image=[
            # "yoda.jpg",
            # "DarthVader.jpeg",
            "Ellipsoid_1680x1680.jpg",
            "ShapesVertex_1680x1680.jpg",
            "HandVertex_1680x1680.jpg",
            "Polygon_1680x1680.jpg",
        ],
    )

    plot_opposite_cells(data_manager, plot_by=["image", "num_cells_per_dim"])
