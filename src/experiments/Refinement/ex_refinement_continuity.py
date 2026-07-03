import functools
import warnings
from collections import defaultdict
from itertools import chain
from pathlib import Path
from typing import Tuple, List

import matplotlib
import networkx as nx
import numpy as np
from matplotlib import pyplot as plt
from tqdm.contrib import itertools

from Results.PaperTest.SchemesTests.SchemesRot.main_script import num_cells_per_dim
from experiments.OtherExperiments.SubcellExperiments.models2compare import quadratic, aero_linear
from experiments.Refinement.ex_refinement_config import experiment_path
from experiments.Refinement.ex_refinement_convergence import get_label4plot
from experiments.Refinement.ex_refinement_tools import do_reconstruction, fit_model, plx_fit_model, obtain_image4error, \
    plx_obtain_image4error, efficient_reconstruction, plx_efficient_reconstruction
from experiments.VizReconstructionUtils import plot_cells, plot_cells_vh_classification_core, \
    plot_cells_not_regular_classification_core, plot_curve_core, draw_cell_borders, draw_numbers, \
    plot_cells_type_of_curve_core
from experiments.global_params import cred
from experiments.tools import calculate_averages_from_curve
from experiments.tools import load_image, calculate_averages_from_image
from experiments.tools4binary_images import plot_reconstruction4img
from lib.CellCreators.CellCreatorBase import REGULAR_CELL_TYPE, CURVE_CELL_TYPE
from lib.Curves.CurveCircle import CurveCircle, CircleParams
from lib.Curves.CurveTrigo import TrigoParams, CurveTrigo
from perplexitylab.experiment_tools import experiment_iterator, concatenate_iterators, define_default_constants, \
    define_default_variables
from perplexitylab.miscellaneous import filter_for_func
from perplexitylab.plot_tools import save_figure

# 1680: 1, 2, 3, 4, 5, 6, 7, 8, 10, 12, 14, 15, 16, 20, 21, 24, 28, 30, 35, 40, 42, 48, 56, 60, 70, 80, 84, 105, 120, 140, 168, 210, 240, 280, 336, 420, 560, 840, 1680
# divisors = [i for i in range(1, n+1) if n % i == 0]

# Reconstruction plot params
matplotlib.rcParams['text.usetex'] = False
curve_color = cred
cmap_reconstruction = "Reds"
cmap_true_image = "Greys_r"
fig_size = (15, 15)

recalculate_obtain_image4error = False
recalculate_fit_model = False


def plot_singular_graph(fig, ax, true_reconstruction, num_cells_per_dim, model, graph,
                        alpha_true_image=0.5,
                        cmap_true_image="Greys_r", draw_mesh=True,
                        trim=((0, 0), (0, 0)), default_linewidth=2,
                        numbers_on=True, vmin=None, vmax=None, labels=True):
    model_resolution = np.array(model.resolution)

    plot_cells(ax, colors=true_reconstruction, mesh_shape=model_resolution, alpha=alpha_true_image,
               cmap=cmap_true_image,
               vmin=np.min(true_reconstruction) if vmin is None else vmin,
               vmax=np.max(true_reconstruction) if vmax is None else vmax,
               labels=labels)

    if draw_mesh:
        draw_cell_borders(
            ax, mesh_shape=num_cells_per_dim,
            refinement=model_resolution // num_cells_per_dim,
            color='black',
            default_linewidth=default_linewidth,
            mesh_style=":"
        )

    ax.set_ylim((model.resolution[1] - trim[0][1] - 0.5, -0.5 + trim[0][0]))
    ax.set_xlim((trim[1][0] - 0.5, model.resolution[0] - trim[1][1] - 0.5))

    draw_numbers(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )

    plot_cells_not_regular_classification_core(ax, model.resolution, model.cells, alpha=0.2, color_border_only=False)
    ax.scatter(*np.array(graph).T[::-1, :], marker=".", color="black")
    ax.plot(*np.array(graph).T[::-1, :], color="black")

    if not numbers_on:
        plt.box(False)
    ax.spines['right'].set_visible(True)
    ax.spines['top'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)


def build_connected_singular_cell_graph(singular_cells_coords: List[Tuple[int, ...]], model, trim):
    f_dist = lambda ci, cj: np.sqrt(np.sum((ci - cj).array ** 2))
    max_dist = 2

    n_cells = len(singular_cells_coords)
    dist = defaultdict(list)
    adj_list = defaultdict(list)
    for i in range(n_cells):
        for j in range(n_cells):
            if i != j:
                d = f_dist(singular_cells_coords[i], singular_cells_coords[j])
                if d < max_dist:
                    if singular_cells_coords[j] not in adj_list[singular_cells_coords[i]]:
                        dist[singular_cells_coords[i]].append(1 * d)
                        adj_list[singular_cells_coords[i]].append(singular_cells_coords[j])
                    if singular_cells_coords[i] not in adj_list[singular_cells_coords[j]]:
                        dist[singular_cells_coords[j]].append(1 * d)
                        adj_list[singular_cells_coords[j]].append(singular_cells_coords[i])
    for i in range(n_cells):
        order = np.argsort(dist[singular_cells_coords[i]])
        adj_list[singular_cells_coords[i]] = [adj_list[singular_cells_coords[i]][o] for o in order]
        dist[singular_cells_coords[i]] = [dist[singular_cells_coords[i]][o] for o in order]

    graph = []
    ix = np.argmin([np.sum(np.abs(c.array)) for c in singular_cells_coords])
    c = singular_cells_coords[ix]
    for i in range(n_cells):
        graph.append(c.tuple)
        for neighbour in adj_list[c]:
            if neighbour.tuple not in graph:
                c = neighbour
                break
        else:
            break
    if f_dist(c, singular_cells_coords[ix]) <= max_dist:  # in case of cyclic border
        graph.append(singular_cells_coords[ix].tuple)

    # filter only the cells that are inside a given region
    graph = [g for g in graph if
             (g[0] >= trim[0][0] - 1) and (g[1] >= trim[0][1] - 1) and
             (g[0] <= model.resolution[1] + trim[1][0]) and (g[1] <= model.resolution[1] + trim[1][1])]

    return graph


def single_experiment_continuity(shape, sub_cell_model, refinement, angle_threshold, num_cells_per_dim,
                                 hash_value=42, sub_discretization2bound_error=10, evaluation_mode=False,
                                 trim=((2, 2), (-2, -2))):
    avg_values = calculate_averages_from_curve(shape, (num_cells_per_dim, num_cells_per_dim))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hash_value, true_reconstruction = plx_obtain_image4error(
            hash_of_preprocess=hash_value, recalculate=recalculate_obtain_image4error,
            shape=shape, num_cells_per_dim=num_cells_per_dim,
            sub_discretization2bound_error=sub_discretization2bound_error,
            avg_values=avg_values, evaluation_mode=evaluation_mode)
        hash_value, model = plx_fit_model(
            hash_of_preprocess=hash_value, recalculate=recalculate_fit_model,
            sub_cell_model=sub_cell_model,
            angle_threshold=angle_threshold,
            refinement=refinement, avg_values=avg_values)

    graph = build_connected_singular_cell_graph(
        singular_cells_coords=[cell.coords for cell in model.cells.values() if cell.CELL_TYPE != REGULAR_CELL_TYPE],
        model=model,
        trim=trim
    )

    return true_reconstruction, model, graph


if __name__ == "__main__":
    # Experiment general params
    noise = 0
    seed = 42
    recalculate_all = False

    # ---------- Experiment list ---------- #
    iterator_builder, info = experiment_iterator(
        experiment_name=Path(__file__).stem,
        constants=define_default_constants(sub_cell_model=None, label=None, angle_threshold=0, reconstruction_factor=1,
                                           recalculate=True),
        variables=define_default_variables(
            num_cells_per_dim=[20, ],
            shape=[
                CurveTrigo(params=TrigoParams(x0=0.5, y0=0.5, amplitude=0.1, frequency=1.)),
                CurveCircle(params=CircleParams(x0=0.511, y0=0.486, radius=0.232))
            ],
            refinement=[1, ]
        ))


    def identifier(info):
        return f"SingularGraph_Img{info.shape}_{info.num_cells_per_dim}x{info.num_cells_per_dim}_{info.label}_Ref{info.refinement}"


    iterators = concatenate_iterators(
        # iterator_builder(sub_cell_model=quadratic, label="AEROS quadratic", refinement=[1, ], angle_threshold=45,
        #                  recalculate=False or recalculate_all),
        iterator_builder(sub_cell_model=aero_linear, label="AEROS linear", refinement=[1, 2, ], angle_threshold=45,
                         recalculate=False or recalculate_all),
        # iterator_builder(sub_cell_model=quadratic, label="AEROS quadratic", refinement=[1, ], angle_threshold=45,
        #                  recalculate=False or recalculate_all),
        # iterator_builder(sub_cell_model=aero_linear, label="AEROS linear", refinement=[1, 2], angle_threshold=45,
        #                  recalculate=False or recalculate_all),
        # iterator_builder(sub_cell_model=elvira, label="ELVIRA", recalculate=False or recalculate_all),
    )

    # ---------- Do experiments ---------- #
    for experiment_info in iterators():
        print("----------------------------------")
        print(identifier(experiment_info))
        true_reconstruction, model, graph = single_experiment_continuity(
            **filter_for_func(single_experiment_continuity, experiment_info._asdict())
        )
        with save_figure(filename=identifier(experiment_info), path=experiment_path, figsize=fig_size,
                         show=False, format="svg") as (fig, ax):
            ax.set_title(get_label4plot(experiment_info.label, experiment_info.refinement))
            plot_singular_graph(
                fig=fig, ax=ax,
                true_reconstruction=true_reconstruction,
                num_cells_per_dim=experiment_info.num_cells_per_dim,
                graph=graph,
                model=model,
                default_linewidth=1.5,
                alpha_true_image=0.15,
                # trim=((1, 1), (2, 2)),
                cmap_true_image=cmap_true_image,
                vmin=0, vmax=1,
                labels=False,
                draw_mesh=True,
                numbers_on=True,
            )
