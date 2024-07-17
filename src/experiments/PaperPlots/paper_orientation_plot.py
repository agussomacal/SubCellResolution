import itertools
import operator
from typing import List

import numpy as np

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.visualization import one_line_iterator, perplex_plot
from experiments.VizReconstructionUtils import plot_cells, draw_cell_borders, plot_specific_cells, \
    SpecialCellsPlotTuple, draw_numbers
from experiments.global_params import cblue, cgreen, cred, image_format
from experiments.tools import calculate_averages_from_image, load_image, \
    singular_cells_mask
from lib.AuxiliaryStructures.Constants import CURVE_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellIterators import iterate_by_reconstruction_error_and_smoothness
from lib.CellOrientators import OrientByGradient

# ========== ========== Names and colors to present ========== ========== #
names_dict = {
    "x": "X independent",
    "y": "Y independent",
    "mixed": "Mixed",
}
ORIENTATION_COLORS = [cgreen, cblue, cred]
ORIENT_NAMES = ["X independent", "Y independent", "Mixed"]


# ========== ========== Experiment definitions ========== ========== #
def experiment(image, num_cells_per_dim, angle_threshold, method="optim", kernel_size=(5, 5)):
    orientator = OrientByGradient(kernel_size=kernel_size, dimensionality=2, method=method,
                                  angle_threshold=angle_threshold)
    image_array = load_image(image)
    avg_values = calculate_averages_from_image(image_array, num_cells_per_dim)
    smoothness_index = singular_cells_mask(avg_values)
    orientations = dict()
    for coords in iterate_by_reconstruction_error_and_smoothness(
            reconstruction_error=np.zeros(np.shape(avg_values)),
            smoothness_index=smoothness_index,
            value=CURVE_CELL, condition=operator.eq):
        o = orientator.get_independent_axis(coords, average_values=avg_values,
                                            indexer=ArrayIndexerNd(avg_values, "cyclic"))
        orientations[coords.tuple] = np.sum(1 + np.array(o))

    return {
        "orientations": orientations,
        "num_singular_cells": np.sum(smoothness_index)
    }


@perplex_plot(legend=False)
@one_line_iterator()
def plot_orientation(fig, ax, image, num_cells_per_dim, orientations,
                     alpha=0.5, cmap="Greys_r", trim=((0, 0), (0, 0)), numbers_on=True,
                     specific_cells: List[SpecialCellsPlotTuple] = [], mesh_linewidth=0):
    image = load_image(image)
    mesh_shape = (num_cells_per_dim, num_cells_per_dim)
    plot_cells(ax, colors=image, mesh_shape=mesh_shape, alpha=alpha, cmap=cmap,
               vmin=np.min(image), vmax=np.max(image))

    directions = [[], [], []]
    for c, o in orientations.items():
        directions[o - 1].append(c)
    special_coords = set(itertools.chain(*[sc.indexes for sc in specific_cells]))
    plot_specific_cells(
        ax=ax,
        mesh_shape=mesh_shape,
        special_cells=[
            SpecialCellsPlotTuple(name=name, indexes=set(coords).difference(special_coords),
                                  color=color, alpha=alpha) for name, color, coords in
            zip(ORIENT_NAMES, ORIENTATION_COLORS, directions)
        ],
        rectangle_mode=False)

    plot_specific_cells(
        ax=ax,
        mesh_shape=mesh_shape,
        special_cells=specific_cells,
        rectangle_mode=False)

    if mesh_linewidth > 0:
        draw_cell_borders(
            ax, mesh_shape=num_cells_per_dim,
            refinement=np.array(mesh_shape) // num_cells_per_dim,
            default_linewidth=mesh_linewidth,
            mesh_style=":",
            color="gray"
        )

    ax.set_ylim((np.array(mesh_shape)[1] - trim[0][1] - 0.5, -0.5 + trim[0][0]))
    ax.set_xlim((trim[1][0] - 0.5, np.array(mesh_shape)[0] - trim[1][1] - 0.5))

    draw_numbers(
        ax, mesh_shape=num_cells_per_dim,
        refinement=np.array(mesh_shape) // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.get_xaxis().set_ticks([])
    ax.get_yaxis().set_ticks([])


data_manager = DataManager(
    path=config.paper_results_path,
    emissions_path=config.results_path,
    name='PaperOrientation',
    format=JOBLIB,
    trackCO2=True,
    country_alpha_code="FR"
)

lab = LabPipeline()

lab.define_new_block_of_functions(
    "experiment_orientation",
    experiment,
    recalculate=False
)
lab.execute(
    data_manager,
    num_cores=15,
    forget=False,
    save_on_iteration=None,
    num_cells_per_dim=[10, 30],  # 60
    image=[
        "batata.jpg",
    ],
    angle_threshold=[
        45
    ],
    method=["sobel"],
    kernel_size=[
        (3, 3),
    ]
)
plot_orientation(
    data_manager,
    path=config.subcell_paper_figures_path,
    method="sobel",
    num_cells_per_dim=10,
    image="batata.jpg",
    angle_threshold=45,
    alpha=0.4,
    format=image_format,
    plot_by=["num_cells_per_dim"],
    numbers_on=False,
    specific_cells=[SpecialCellsPlotTuple(name="SpecialCell", indexes=[(8, 8)],
                                          color=cred, alpha=0.5),
                    ],
    mesh_linewidth=1,
    trim=((0, 0), (0, 0)),
)

plot_orientation(
    data_manager,
    path=config.subcell_paper_figures_path,
    method="sobel",
    num_cells_per_dim=30,
    image="batata.jpg",
    angle_threshold=45,
    alpha=0.4,
    format=image_format,
    plot_by=["num_cells_per_dim"],
    numbers_on=False,
    mesh_linewidth=1,
    trim=((0, 0), (0, 0)),
)
