import time

import numpy as np
import seaborn as sns
from tqdm import tqdm

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot, one_line_iterator, perplex_plot
from experiments.VizReconstructionUtils import plot_cells, draw_cell_borders, plot_cells_identity, \
    plot_cells_vh_classification_core, plot_cells_not_regular_classification_core, plot_curve_core
from experiments.subcell_paper.global_params import cpink, corange, cyellow, \
    cblue, cgreen, runsinfo, EVALUATIONS, cpurple, cred, ccyan
from experiments.subcell_paper.models2compare import upwind, elvira, aero_linear, quadratic, aero_qelvira_vertex, \
    aero_lq, qelvira
from experiments.subcell_paper.tools import calculate_averages_from_image, load_image, \
    reconstruct, singular_cells_mask, get_reconstruction_error
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import REGULAR_CELL_TYPE
from lib.SubCellScheme import SubCellScheme

SAVE_EACH = 5

# ========== ========== Names and colors to present ========== ========== #
names_dict = {
    "upwind": "Up Wind",
    "elvira": "ELVIRA",
    "aero_linear": "AERO-Linear",
    "quadratic": "AERO-Quadratic",
    "aero_lq": "AERO-Linear-Quadratic",
    "qelvira": "ELVIRA AERO-Quadratic",
    "aero_qelvira_vertex": "AERO-Quadratic Vertex",
    "aero_qelvira_vertex45": "AERO-Quadratic Orient Vertex",
}
model_color = {
    "upwind": cpink,
    "elvira": corange,
    "aero_linear": cyellow,
    "quadratic": cblue,
    "aero_lq": cpurple,
    "qelvira": cred,
    "aero_qelvira_vertex": cgreen,
    "aero_qelvira_vertex45": ccyan,
}
names_dict = {k: names_dict[k] for k in model_color.keys()}

runsinfo.append_info(
    **{k.replace("_", "-"): v for k, v in names_dict.items()}
)


# ========== ========== Experiment definitions ========== ========== #
def calculate_true_solution(image, num_cells_per_dim, velocity, ntimes):
    image = load_image(image)
    pixels_per_cell = np.array(np.shape(image)) / num_cells_per_dim
    velocity_in_pixels = np.array(pixels_per_cell * np.array(velocity), dtype=int)
    assert np.all(velocity_in_pixels == pixels_per_cell * np.array(velocity))

    true_solution = []
    true_reconstruction = []
    for i in range(ntimes + 1):
        if i % SAVE_EACH == 0:
            true_reconstruction.append(image.copy())
        true_solution.append(calculate_averages_from_image(image, num_cells_per_dim))
        image = np.roll(image, velocity_in_pixels)

    return {
        "true_solution": true_solution,
        "true_reconstruction": true_reconstruction
    }


def fit_model(subcell_reconstruction):
    def decorated_func(image, noise, num_cells_per_dim, reconstruction_factor, velocity, ntimes, true_solution):
        image_array = load_image(image)
        avg_values = calculate_averages_from_image(image_array, num_cells_per_dim)
        np.random.seed(42)
        avg_values = avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)

        model = SubCellScheme(name=subcell_reconstruction.__name__, subcell_reconstructor=subcell_reconstruction(),
                              min_value=0, max_value=1)

        # finite volume solver evolution
        t0 = time.time()
        solution, all_cells = model.evolve(
            init_average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"),
            velocity=np.array(velocity), ntimes=ntimes,
            interface_oracle=singular_cells_mask(true_solution)
        )
        t_fit = time.time() - t0

        all_cells = [cell for i, cell in enumerate(all_cells) if i % SAVE_EACH == 0]

        # do reconstruction
        t0 = time.time()
        reconstruction = []
        for i, cells in tqdm(enumerate(all_cells), desc="Reconstruction."):
            reconstruction.append(reconstruct(image_array, cells, model.resolution, reconstruction_factor,
                                              do_evaluations=EVALUATIONS))
        t_reconstruct = time.time() - t0

        return {
            "model": model,
            "time_to_fit": t_fit,
            "reconstruction": reconstruction,
            "cells": all_cells,
            "solution": solution,
            "time_to_reconstruct": t_reconstruct
        }

    # need to change the name so the lab experiment saves the correct name and not the uniformly "decorated_func"
    # the other option is to pass to the block the name we wish to associate to the function.
    decorated_func.__name__ = subcell_reconstruction.__name__
    return decorated_func


# ========== ========== Plots definitions ========== ========== #
@perplex_plot()
@one_line_iterator
def plot_time_i(fig, ax, true_solution, solution, num_cells_per_dim, model, i=0, alpha=0.5, cmap="Greys_r",
                trim=((0, 0), (0, 0)),
                numbers_on=True, error=False):
    model_resolution = np.array(model.resolution)
    colors = (solution[i] - true_solution[i]) if error else solution[i]
    plot_cells(ax, colors=colors, mesh_shape=model_resolution, alpha=alpha, cmap=cmap,
               vmin=np.min(true_solution), vmax=np.max(true_solution))

    draw_cell_borders(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )
    ax.set_xlim((-0.5 + trim[0][0], model_resolution[0] - trim[0][1] - 0.5))
    ax.set_ylim((model_resolution[1] - trim[1][0] - 0.5, trim[1][1] - 0.5))


@perplex_plot()
@one_line_iterator
def plot_reconstruction_time_i(fig, ax, true_reconstruction, num_cells_per_dim, model, reconstruction, cells, i=0,
                               alpha=0.5,
                               plot_original_image=True,
                               difference=False, plot_curve=True, plot_curve_winner=False, plot_vh_classification=True,
                               plot_singular_cells=True, cmap="magma", trim=((0, 0), (0, 0)), numbers_on=True):
    model_resolution = np.array(model.resolution)
    image = true_reconstruction[i]

    if plot_original_image:
        plot_cells(ax, colors=image, mesh_shape=model_resolution, alpha=alpha, cmap="Greys_r",
                   vmin=np.min(image), vmax=np.max(image))

    if difference:
        # TODO: should be the evaluations not the averages.
        image = calculate_averages_from_image(image, num_cells_per_dim=np.shape(reconstruction))
        plot_cells(ax, colors=reconstruction[i] - image, mesh_shape=model.resolution, alpha=alpha, cmap=cmap, vmin=-1,
                   vmax=1)
    else:
        plot_cells(ax, colors=reconstruction[i], mesh_shape=model.resolution, alpha=alpha, cmap=cmap, vmin=-1, vmax=1)

    if plot_curve:
        if plot_curve_winner:
            plot_cells_identity(ax, model.resolution, cells[i], alpha=0.8)
            # plot_cells_type_of_curve_core(ax, model.resolution, model.cells, alpha=0.8)
        elif plot_vh_classification:
            plot_cells_vh_classification_core(ax, model.resolution, cells[i], alpha=0.8)
        elif plot_singular_cells:
            plot_cells_not_regular_classification_core(ax, model.resolution, cells[i], alpha=0.8)
        plot_curve_core(ax, curve_cells=[cell for cell in cells[i].values() if
                                         cell.CELL_TYPE != REGULAR_CELL_TYPE])

    draw_cell_borders(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )
    ax.set_xlim((-0.5 + trim[0][0], model.resolution[0] - trim[0][1] - 0.5))
    ax.set_ylim((model.resolution[1] - trim[1][0] - 0.5, trim[1][1] - 0.5))


# ========== ========== Error definitions ========== ========== #
scheme_error = lambda image, true_solution, solution: np.mean(
    np.abs((np.array(solution[1:]) - np.array(true_solution[1:]))), axis=(1, 2))
scheme_reconstruction_error = lambda true_reconstruction, reconstruction, reconstruction_factor: np.array([
    get_reconstruction_error(tr_i, reconstruction=r_i, reconstruction_factor=reconstruction_factor)
    for r_i, tr_i in zip(reconstruction, true_reconstruction)])

if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='Schemes',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()

    lab.define_new_block_of_functions(
        "ground_truth",
        calculate_true_solution,
        recalculate=False
    )

    lab.define_new_block_of_functions(
        "models",
        *map(fit_model, [
            upwind,
            # elvira_oriented,
            elvira,
            aero_linear,
            # aero_linear_oriented,
            quadratic,
            qelvira,
            # quadratic_oriented,
            aero_lq,
            aero_qelvira_vertex,
            NamedPartial(aero_qelvira_vertex, angle_threshold=45).add_sufix_to_name(45),
            # obera_aero_lq_vertex,
        ]),
        recalculate=False
    )

    ntimes = 120
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        refinement=[1],
        ntimes=[ntimes],
        velocity=[(0, 1 / 4)],
        num_cells_per_dim=[30],  # 60
        noise=[0],
        image=[
            # "yoda.jpg",
            # "DarthVader.jpeg",
            # "Ellipsoid_1680x1680.jpg",
            "ShapesVertex_1680x1680.jpg",
            "HandVertex_1680x1680.jpg",
            "Polygon_1680x1680.jpg",
        ],
        reconstruction_factor=[5],
        # reconstruction_factor=[1],
    )

    generic_plot(data_manager,
                 name="ErrorInTime",
                 format=".pdf",
                 path=config.subcell_paper_figures_path,
                 x="times", y="scheme_error", label="method", plot_by=["num_cells_per_dim", "image"],
                 # models=["elvira", "quadratic"],
                 times=lambda ntimes: np.arange(1, ntimes + 1),
                 scheme_error=scheme_error,
                 plot_func=NamedPartial(
                     sns.lineplot, marker="o", linestyle="--",
                     palette={v: model_color[k] for k, v in names_dict.items()}
                 ),
                 models=list(model_color.keys()),
                 method=lambda models: names_dict[models],
                 log="y",
                 )
    #
    # new_times = np.array([1, 7, 13, 19, 26, 32, 38, 44, 51, 57, 63, 69, 76, 82, 88, 94, 101, 107, 113, 119])
    generic_plot(data_manager,
                 name="ReconstructionErrorInTime",
                 format=".pdf",
                 path=config.subcell_paper_figures_path,
                 x="times", y="scheme_error", label="method", plot_by=["num_cells_per_dim", "image"],
                 # models=["elvira", "quadratic"],
                 times=lambda ntimes: np.arange(0, ntimes+1, SAVE_EACH),
                 # times=lambda ntimes: new_times,
                 scheme_error=scheme_reconstruction_error,
                 plot_func=NamedPartial(
                     sns.lineplot, marker="o", linestyle="--",
                     palette={v: model_color[k] for k, v in names_dict.items()}
                 ),
                 log="y",
                 models=list(model_color.keys()),
                 method=lambda models: names_dict[models],
                 )

    generic_plot(data_manager,
                 name="Time2Fit",
                 x="image", y="time_to_fit", label="method", plot_by=["num_cells_per_dim", "image"],
                 # models=["elvira", "quadratic"],
                 times=lambda ntimes: np.arange(ntimes),
                 plot_func=NamedPartial(
                     sns.barplot,
                     palette={v: model_color[k] for k, v in names_dict.items()}
                 ),
                 log="y",
                 models=list(model_color.keys()),
                 method=lambda models: names_dict[models],
                 )

    for i in range(ntimes):
        plot_time_i(data_manager, folder="Solution", name=f"Time{i}", i=i, alpha=0.8, cmap="viridis",
                    trim=((0, 0), (0, 0)), folder_by=['image', 'num_cells_per_dim'],
                    plot_by=[],
                    axes_by=["method"],
                    models=list(model_color.keys()),
                    method=lambda models: names_dict[models],
                    numbers_on=True, error=True)

    for i in range(0, ntimes + 1, SAVE_EACH):
        plot_reconstruction_time_i(
            data_manager,
            i=i // SAVE_EACH,
            name=f"Reconstruction{i}",
            folder='Reconstruction',
            folder_by=['image', 'num_cells_per_dim'],
            plot_by=[],
            axes_by=["method"],
            models=list(model_color.keys()),
            method=lambda models: names_dict[models],
            axes_xy_proportions=(15, 15),
            difference=False,
            plot_curve=True,
            plot_curve_winner=False,
            plot_vh_classification=False,
            plot_singular_cells=False,
            plot_original_image=True,
            numbers_on=True,
            plot_again=True,
            num_cores=1,
        )
