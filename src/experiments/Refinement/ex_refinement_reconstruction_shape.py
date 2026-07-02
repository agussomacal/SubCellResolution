import warnings
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib import pyplot as plt

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
from lib.CellCreators.CellCreatorBase import REGULAR_CELL_TYPE
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


def plot_reconstruction4shape(fig, ax, true_reconstruction, num_cells_per_dim, model, reconstruction,
                              alpha=0.5, alpha_true_image=0.5, alpha_vh=0.2, difference=False, plot_curve=True,
                              plot_curve_winner=False,
                              plot_vh_classification=True, plot_singular_cells=True, cmap="viridis",
                              cmap_true_image="Greys_r", draw_mesh=True,
                              trim=((0, 1), (0, 1)), default_linewidth=2, color_border_only=False,
                              numbers_on=True, vmin=None, vmax=None, labels=True, curve_color=None):
    model_resolution = np.array(model.resolution)

    if alpha_true_image > 0:
        plot_cells(ax, colors=true_reconstruction, mesh_shape=model_resolution, alpha=alpha_true_image,
                   cmap=cmap_true_image,
                   vmin=np.min(true_reconstruction) if vmin is None else vmin,
                   vmax=np.max(true_reconstruction) if vmax is None else vmax,
                   labels=labels)

    if alpha > 0:
        if difference:
            d = reconstruction - true_reconstruction
            plot_cells(ax, colors=d, mesh_shape=model_resolution,
                       alpha=alpha, cmap=cmap,
                       vmin=np.min(d) if vmin is None else vmin,
                       vmax=np.max(d) if vmax is None else vmax,
                       labels=labels)
        else:
            plot_cells(ax, colors=reconstruction, mesh_shape=model_resolution,
                       alpha=alpha, cmap=cmap,
                       vmin=np.min(reconstruction) if vmin is None else vmin,
                       vmax=np.max(reconstruction) if vmax is None else vmax,
                       labels=labels)

    if plot_curve:
        if plot_curve_winner:
            # plot_cells_identity(ax, model.resolution, model.cells, alpha=0.8, color_dict=winner_color_dict)
            plot_cells_type_of_curve_core(ax, model.resolution, model.cells, alpha=0.8)
        elif plot_vh_classification:
            plot_cells_vh_classification_core(ax, model.resolution, model.cells, alpha=alpha_vh,
                                              color_border_only=color_border_only)
        elif plot_singular_cells:
            plot_cells_not_regular_classification_core(ax, model.resolution, model.cells, alpha=0.8)
        plot_curve_core(ax, curve_cells=[cell for cell in model.cells.values() if
                                         cell.CELL_TYPE != REGULAR_CELL_TYPE],
                        default_linewidth=default_linewidth * 1.5,
                        color=curve_color)

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

    if not numbers_on:
        plt.box(False)


def single_experiment_reconstruction_shape(shape, sub_cell_model, refinement, angle_threshold, num_cells_per_dim,
                                           sub_discretization2bound_error, recalculate_inner_funcs, hash_value=42,
                                           evaluation_mode=False, do_efficient_reconstruction=True):
    avg_values = calculate_averages_from_curve(shape, (num_cells_per_dim, num_cells_per_dim))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        hash_value, true_reconstruction = plx_obtain_image4error(
            hash_of_preprocess=hash_value, recalculate=recalculate_obtain_image4error,
            shape=shape, num_cells_per_dim=num_cells_per_dim,
            sub_discretization2bound_error=sub_discretization2bound_error,
            avg_values=avg_values, evaluation_mode=evaluation_mode)
        hash_value, model = plx_fit_model(
            hash_of_preprocess=hash_value, recalculate=recalculate_inner_funcs,
            sub_cell_model=sub_cell_model,
            angle_threshold=angle_threshold,
            refinement=refinement, avg_values=avg_values)
    if do_efficient_reconstruction:
        _, reconstruction = plx_efficient_reconstruction(
            hash_of_preprocess=hash_value, recalculate=recalculate_inner_funcs,
            model=model, avg_values=avg_values,
            sub_discretization2bound_error=sub_discretization2bound_error,
            refinement=refinement, evaluation_mode=evaluation_mode)
    else:
        _, reconstruction = do_reconstruction(hash_of_preprocess=hash_value,
                                              recalculate=experiment_info.recalculate,
                                              image=true_reconstruction, model=model,
                                              reconstruction_factor=experiment_info.reconstruction_factor,
                                              do_evaluations=experiment_info.do_evaluations)
    return reconstruction, true_reconstruction, model


if __name__ == "__main__":
    # Experiment general params
    noise = 0
    seed = 42
    recalculate_all = False

    # ---------- Experiment list ---------- #
    iterator_builder, info = experiment_iterator(
        experiment_name=Path(__file__).stem,
        constants=define_default_constants(sub_cell_model=None, label=None, angle_threshold=0, reconstruction_factor=1,
                                           sub_discretization2bound_error=8 * 3, recalculate=False,
                                           recalculate_inner_funcs=False, evaluation_mode=False),
        variables=define_default_variables(
            num_cells_per_dim=[20, ],
            shape=[CurveTrigo(params=TrigoParams(x0=0.5, y0=0.5, amplitude=0.1, frequency=1.)),
                   CurveCircle(params=CircleParams(x0=0.511, y0=0.486, radius=0.232))],
            refinement=[1, 2]
        ))


    def identifier(info):
        return f"Img{info.shape}_{info.num_cells_per_dim}x{info.num_cells_per_dim}_{info.label}_Ref{info.refinement}"


    iterators = concatenate_iterators(
        iterator_builder(sub_cell_model=quadratic, label="AEROS quadratic", refinement=[1, ], angle_threshold=45,
                         recalculate=False or recalculate_all),
        iterator_builder(sub_cell_model=aero_linear, label="AEROS linear", refinement=[1, 2, 3], angle_threshold=45,
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
        reconstruction, true_reconstruction, model = single_experiment_reconstruction_shape(
            do_efficient_reconstruction=True,
            **filter_for_func(single_experiment_reconstruction_shape, experiment_info._asdict())
        )
        with save_figure(filename=identifier(experiment_info), path=experiment_path, figsize=fig_size,
                         show=False, format="svg") as (fig, ax):
            ax.set_title(get_label4plot(experiment_info.label, experiment_info.refinement))
            plot_reconstruction4shape(
                fig=fig, ax=ax,
                true_reconstruction=true_reconstruction,
                num_cells_per_dim=experiment_info.num_cells_per_dim,
                model=model,
                reconstruction=reconstruction,
                difference=False,
                plot_curve=True,
                plot_curve_winner=False,
                plot_vh_classification=True,
                plot_singular_cells=False,
                default_linewidth=1.5,
                alpha_true_image=0.15,
                alpha=0,
                alpha_vh=0.2,
                # trim=((1, 1), (2, 2)),
                cmap=cmap_reconstruction,
                cmap_true_image=cmap_true_image,
                curve_color=curve_color,
                color_border_only=False,
                vmin=0, vmax=1,
                labels=False,
                draw_mesh=True,
                numbers_on=True,
            )
