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
from experiments.global_params import cblue, cgreen, cred
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


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='Orientation',
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
    num_cells_per_dim = [5, 10, 15, 20, 28, 30, 42]  # 60
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        num_cells_per_dim=num_cells_per_dim,  # 60
        image=[
            # "yoda.jpg",
            # "DarthVader.jpeg",
            "batata.jpg",
            # "mafalda.jpg",
            # "Ellipsoid_1680x1680.jpg",
            # "ShapesVertex_1680x1680.jpg",
            # "HandVertex_1680x1680.jpg",
            # "Polygon_1680x1680.jpg",
        ],
        angle_threshold=[
            # 30,
            45
        ],
        method=[
            "sobel",
            "optim"
        ],
        kernel_size=[
            (3, 3),
            (5, 5),
        ]
    )
    # lab.execute(
    #     data_manager,
    #     num_cores=15,
    #     forget=False,
    #     save_on_iteration=None,
    #     num_cells_per_dim=[10, 15, 20, 28, 30, 42],  # 60
    #     image=[
    #         # "yoda.jpg",
    #         # "DarthVader.jpeg",
    #         "Ellipsoid_1680x1680.jpg",
    #         "ShapesVertex_1680x1680.jpg",
    #         "HandVertex_1680x1680.jpg",
    #         "Polygon_1680x1680.jpg",
    #     ],
    #     angle_threshold=[
    #         30,
    #         45
    #     ],
    #     method=["optim"],
    #     kernel_size=[
    #         (3, 3),
    #         (5, 5)
    #     ]
    # )

    plot_orientation(
        data_manager,
        # method="sobel",
        num_cells_per_dim=num_cells_per_dim,
        # image="mafalda.jpg",
        image="batata.jpg",
        angle_threshold=45,
        alpha=0.5,
        plot_by=["num_cells_per_dim"],
        axes_by=["method", "kernel_size"],
        # format=".pdf",
        # trim=((1, 1), (1, 1)),
        numbers_on=False,
    )
    #
    # generic_plot(
    #     data_manager,
    #     x="num_cells_per_dim", y="prop_mixed_cells", label="orientator",
    #     orientator=lambda kernel_size, angle_threshold, method: f"{method}: {kernel_size} - {angle_threshold}º",
    #     prop_mixed_cells=lambda orientations, num_singular_cells: list(orientations.values()).count(
    #         3) / num_singular_cells,
    #     axes_by=["image"]
    # )
    # plot_orientation(data_manager, axes_by=["angle_threshold", "kernel_size", "method"],
    #                  plot_by=["image", "num_cells_per_dim"])
