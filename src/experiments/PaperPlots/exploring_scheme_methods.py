import time

import numpy as np
import seaborn as sns
from scipy import ndimage
from tqdm import tqdm

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot, one_line_iterator, perplex_plot
from experiments.VizReconstructionUtils import plot_cells, draw_cell_borders, plot_cells_identity, \
    plot_cells_vh_classification_core, plot_cells_not_regular_classification_core, plot_curve_core, draw_numbers
from experiments.global_params import cpink, corange, cyellow, \
    cblue, cgreen, runsinfo, EVALUATIONS, cpurple, cred, ccyan, cgray, RESOLUTION_FACTOR, num_cores
from experiments.PaperPlots.models2compare import qelvira, upwind
from experiments.tools import calculate_averages_from_image, load_image, \
    reconstruct, singular_cells_mask, get_reconstruction_error
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE, VERTEX_CELL_TYPE
from lib.CellCreators.LearningFluxRegularCellCreator import CellLearnedFlux
from lib.SubCellScheme import SubCellScheme

SAVE_EACH = 1

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
    "nn_fluxlines": "NN Flux Lines",
    "nn_fluxquadratics": "NN Flux Quadratics",
    "nn_fluxlinesquadratics": "NN Flux Lines-Quadratics",
    "qelvira_kml": "ML-Kernel ELVIRA AERO-Quadratic",
    "ml_vql": "ML ELVIRA AERO-Quadratic Vertex",
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
    "qelvira_kml": cgray,
    "nn_fluxlines": "forestgreen",
    "nn_fluxquadratics": corange,
    "nn_fluxlinesquadratics": ccyan,
    "ml_vql": cblue,
}
names_dict = {k: names_dict[k] for k in model_color.keys()}

runsinfo.append_info(
    **{k.replace("_", "-"): v for k, v in names_dict.items()}
)


def get_velocity_field(velocity, num_cells_per_dim, center=np.zeros(2), angular_velocity=0):
    """

    :param velocity:
    :param num_cells_per_dim:
    :param center: measured [0, 1]
    :param angular_velocity:
    :return:
    velocity:
    rot_velocity: num_cells_per_dim x num_cells_per_dim x 2
    """
    h = 1.0 / num_cells_per_dim
    if angular_velocity != 0:
        xy = h * (1 / 2 + np.array(np.meshgrid(range(num_cells_per_dim), range(num_cells_per_dim), indexing="ij")))
        r = (xy - np.array(center)[:, np.newaxis, np.newaxis])  # radius
        # rotation of 90º angle
        r = r[[1, 0]]  # reflection around 45º line
        r[0] = -r[0]  # reflection around x=0
        new_angular_velocity = angular_velocity / np.max(np.abs(r))
        rot_velocity = np.transpose(r, axes=(1, 2, 0)) * new_angular_velocity
    else:
        new_angular_velocity = 0
        rot_velocity = 0
    return np.array(velocity), rot_velocity, new_angular_velocity * h


# ========== ========== Experiment definitions ========== ========== #
def get_velocity4experiments(num_cells_per_dim, velocity, angular_velocity):
    center = np.ones(2) / 2
    velocity, rot_velocity, new_angular_velocity = get_velocity_field(velocity, num_cells_per_dim, center,
                                                                      angular_velocity)
    return velocity, rot_velocity, center, new_angular_velocity


def calculate_true_solution(image, num_cells_per_dim, velocity, ntimes, angular_velocity, num_screenshots):
    times2save = np.linspace(0, ntimes-1, num_screenshots, dtype=int)
    image = load_image(image)
    pixels_per_cell = np.array(np.shape(image)) / num_cells_per_dim

    velocity, _, _, new_angular_velocity = get_velocity4experiments(num_cells_per_dim, velocity, angular_velocity)
    velocity_in_pixels = np.array(pixels_per_cell * np.array(velocity), dtype=int)
    assert np.all(velocity_in_pixels == pixels_per_cell * np.array(velocity))

    original_image = image.copy()
    true_solution = []
    true_reconstruction = []
    for i in range(ntimes + 1):
        if i in times2save:
            true_reconstruction.append(image.copy())
        true_solution.append(calculate_averages_from_image(image, num_cells_per_dim))
        if not np.any(velocity):
            image = ndimage.rotate(original_image, 180 * new_angular_velocity * (i + 1) / np.pi, mode="mirror",
                                   reshape=False)
        elif new_angular_velocity == 0:
            image = np.roll(original_image, velocity_in_pixels * (i + 1))
        else:
            image = np.roll(image, velocity_in_pixels / 2)
            image = ndimage.rotate(image, 180 * new_angular_velocity / np.pi, mode="mirror", reshape=False)
            image = np.roll(image, velocity_in_pixels / 2)
        # threshold to binarize again
        image[image < 0.5] = 0
        image[image >= 0.5] = 1

    return {
        "true_solution": true_solution,
        "true_reconstruction": true_reconstruction
    }


def fit_model(subcell_reconstruction):
    def decorated_func(image, noise, num_cells_per_dim, reconstruction_factor, velocity, ntimes, true_solution,
                       angular_velocity, num_screenshots):
        image_array = load_image(image)
        avg_values = calculate_averages_from_image(image_array, num_cells_per_dim)
        np.random.seed(42)
        avg_values = avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)

        model = SubCellScheme(name=subcell_reconstruction.__name__, subcell_reconstructor=subcell_reconstruction(),
                              min_value=0, max_value=1)

        # pixels_per_cell = np.array(np.shape(image_array)) / num_cells_per_dim
        # angular_velocity = get_relative_angular_velocity(num_cells_per_dim, angular_velocity)*pixels_per_cell[0]
        velocity, rot_velocity, _, _ = get_velocity4experiments(num_cells_per_dim, velocity, angular_velocity)
        velocity = rot_velocity + velocity
        # finite volume solver evolution
        t0 = time.time()
        solution, all_cells = model.evolve(
            init_average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"),
            velocity=np.array(velocity), ntimes=ntimes,
            interface_oracle=singular_cells_mask(true_solution)
        )
        t_fit = time.time() - t0

        # retain only certain times for reconstruction
        times2save = np.linspace(0, ntimes-1, num_screenshots, dtype=int)
        all_cells = [cell for i, cell in enumerate(all_cells) if i in times2save]

        # do reconstruction
        t0 = time.time()
        reconstruction = []
        for i, cells in tqdm(enumerate(all_cells), desc="Reconstruction."):
            if CellLearnedFlux in map(type, cells.values()):
                print("Flux method, no reconstruction.")
                reconstruction = None
                all_cells = None  # otherwise it does not pickle
                break
            reconstruction.append(reconstruct(image_array, cells, model.resolution, reconstruction_factor,
                                              do_evaluations=EVALUATIONS))
        t_reconstruct = time.time() - t0

        return {
            "resolution": model.resolution,
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
@one_line_iterator()
def plot_time_i(fig, ax, true_solution, solution, num_cells_per_dim, i=0, alpha=0.5, cmap="Greys_r",
                trim=((0, 0), (0, 0)),
                numbers_on=True, error=False, draw_mesh=True):
    model_resolution = np.array([num_cells_per_dim, num_cells_per_dim])
    colors = (solution[i] - true_solution[i]) if error else solution[i]
    plot_cells(ax, colors=colors, mesh_shape=model_resolution, alpha=alpha, cmap=cmap,
               vmin=np.min(true_solution), vmax=np.max(true_solution))

    if draw_mesh:
        draw_cell_borders(
            ax, mesh_shape=num_cells_per_dim,
            refinement=model_resolution // num_cells_per_dim,
        )

    ax.set_ylim((model_resolution[1] - trim[0][1] - 0.5, -0.5 + trim[0][0]))
    ax.set_xlim((trim[1][0] - 0.5, model_resolution[0] - trim[1][1] - 0.5))

    draw_numbers(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )


@perplex_plot(legend=False)
@one_line_iterator()
def plot_reconstruction_time_i(fig, ax, true_reconstruction, num_cells_per_dim, resolution, reconstruction, cells, i=0,
                               alpha=0.5, alpha_true_image=0.5, difference=False, plot_curve=True,
                               plot_curve_winner=False,
                               plot_vh_classification=True, plot_singular_cells=True, cmap="viridis",
                               cmap_true_image="Greys_r", draw_mesh=True,
                               trim=((0, 1), (0, 1)),
                               numbers_on=True, vmin=None, vmax=None, labels=True):
    model_resolution = np.array(resolution)
    image = true_reconstruction[i]

    if alpha_true_image > 0:
        plot_cells(ax, colors=image, mesh_shape=model_resolution, alpha=alpha_true_image, cmap=cmap_true_image,
                   vmin=np.min(image) if vmin is None else vmin,
                   vmax=np.max(image) if vmax is None else vmax,
                   labels=labels)

    if difference:
        d = reconstruction[i] - image
        plot_cells(ax, colors=d, mesh_shape=model_resolution,
                   alpha=alpha, cmap=cmap,
                   vmin=np.min(d) if vmin is None else vmin,
                   vmax=np.max(d) if vmax is None else vmax,
                   labels=labels)
    else:
        plot_cells(ax, colors=reconstruction[i], mesh_shape=model_resolution,
                   alpha=alpha, cmap=cmap,
                   vmin=np.min(reconstruction[i]) if vmin is None else vmin,
                   vmax=np.max(reconstruction[i]) if vmax is None else vmax,
                   labels=labels)

    if plot_curve:
        if plot_curve_winner:
            plot_cells_identity(ax, model_resolution, cells[i], alpha=0.8, color_dict=plot_curve_winner)
            # plot_cells_type_of_curve_core(ax, model.resolution, model.cells, alpha=0.8)
        elif plot_vh_classification:
            plot_cells_vh_classification_core(ax, model_resolution, cells[i], alpha=0.8)
        elif plot_singular_cells:
            plot_cells_not_regular_classification_core(ax, model_resolution, cells[i], alpha=0.8)
        plot_curve_core(ax, curve_cells=[cell for cell in cells[i].values() if
                                         cell.CELL_TYPE in [CURVE_CELL_TYPE, VERTEX_CELL_TYPE]],
                        default_linewidth=3.5 * 20 / num_cells_per_dim)

    if draw_mesh:
        draw_cell_borders(
            ax, mesh_shape=num_cells_per_dim,
            refinement=model_resolution // num_cells_per_dim,
            color='black',
            default_linewidth=2,
            mesh_style=":"
        )

    ax.set_ylim((model_resolution[1] - trim[0][1] - 0.5, -0.5 + trim[0][0]))
    ax.set_xlim((trim[1][0] - 0.5, model_resolution[0] - trim[1][1] - 0.5))

    draw_numbers(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )


# ========== ========== Error definitions ========== ========== #
scheme_error = lambda image, true_solution, solution: np.mean(
    np.abs((np.array(solution[1:]) - np.array(true_solution[1:]))), axis=(1, 2))
scheme_reconstruction_error = lambda true_reconstruction, reconstruction, reconstruction_factor: np.array([
    get_reconstruction_error(tr_i, reconstruction=r_i, reconstruction_factor=reconstruction_factor)
    for r_i, tr_i in zip(reconstruction, true_reconstruction)]) if reconstruction is not None else None

if __name__ == "__main__":
    # Tests
    num_cells_per_dim = 50
    velocity, rot_velocity = get_velocity_field(velocity=(0, 0), num_cells_per_dim=num_cells_per_dim,
                                                center=np.zeros(2),
                                                angular_velocity=1)
    velocity = velocity + rot_velocity
    assert np.allclose(np.max(np.sqrt(np.sum(velocity ** 2, axis=-1))), np.sqrt(2), 10)

    import numpy as np
    import matplotlib.pyplot as plt

    num_cells_per_dim = 10
    velocity, rot_velocity, center = get_velocity4experiments(num_cells_per_dim, velocity=(0, 0), angular_velocity=1)
    velocity = velocity + rot_velocity
    x, y = np.meshgrid(np.linspace(0, 1, num_cells_per_dim), np.linspace(0, 1, num_cells_per_dim), indexing="ij")
    plt.quiver(x, y, velocity[:, :, 0], velocity[:, :, 1])
    plt.show()
    print(velocity[0, 0])
    print(velocity[-1, 0])
    print(velocity[-1, -1])
    print(velocity[0, -1])
