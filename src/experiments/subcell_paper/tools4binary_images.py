import time

import numpy as np

from PerplexityLab.visualization import perplex_plot, one_line_iterator
from experiments.VizReconstructionUtils import plot_cells, plot_cells_identity, plot_cells_vh_classification_core, \
    plot_cells_not_regular_classification_core, plot_curve_core, draw_cell_borders
from experiments.subcell_paper.global_params import EVALUATIONS
from experiments.subcell_paper.tools import load_image, calculate_averages_from_image, reconstruct
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import REGULAR_CELL_TYPE


def fit_model(sub_cell_model):
    def decorated_func(image, noise, num_cells_per_dim, reconstruction_factor, refinement, angle_threshold=30):
        image = load_image(image)
        avg_values = calculate_averages_from_image(image, num_cells_per_dim)
        np.random.seed(42)
        avg_values = avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)

        model = sub_cell_model(refinement=refinement, angle_threshold=angle_threshold)

        t0 = time.time()
        model.fit(average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"))
        t_fit = time.time() - t0

        t0 = time.time()
        reconstruction = reconstruct(image, model.cells, model.resolution, reconstruction_factor,
                                     do_evaluations=EVALUATIONS)
        t_reconstruct = time.time() - t0

        return {
            "model": model,
            "time_to_fit": t_fit,
            "reconstruction": reconstruction,
            "time_to_reconstruct": t_reconstruct
        }

    # need to change the name so the lab experiment saves the correct name and not the uniformly "decorated_func"
    # the other option is to pass to the block the name we wish to associate to the function.
    decorated_func.__name__ = sub_cell_model.__name__
    return decorated_func


@perplex_plot()
@one_line_iterator
def plot_reconstruction(fig, ax, image, num_cells_per_dim, model, reconstruction, alpha=0.5,
                        plot_original_image=True,
                        difference=False, plot_curve=True, plot_curve_winner=False, plot_vh_classification=True,
                        plot_singular_cells=True, cmap="magma", trim=((0, 0), (0, 0)), numbers_on=True,
                        winner_color_dict=None):
    model_resolution = np.array(model.resolution)
    image = load_image(image)

    if plot_original_image:
        plot_cells(ax, colors=image, mesh_shape=model_resolution, alpha=alpha, cmap="Greys_r",
                   vmin=np.min(image), vmax=np.max(image))

    if difference:
        # TODO: should be the evaluations not the averages.
        image = calculate_averages_from_image(image, num_cells_per_dim=np.shape(reconstruction))
        plot_cells(ax, colors=reconstruction - image, mesh_shape=model.resolution, alpha=alpha, cmap=cmap, vmin=-1,
                   vmax=1)
    else:
        plot_cells(ax, colors=reconstruction, mesh_shape=model.resolution, alpha=alpha, cmap=cmap, vmin=-1, vmax=1)

    if plot_curve:
        if plot_curve_winner:
            plot_cells_identity(ax, model.resolution, model.cells, alpha=0.8, color_dict=winner_color_dict)
            # plot_cells_type_of_curve_core(ax, model.resolution, model.cells, alpha=0.8)
        elif plot_vh_classification:
            plot_cells_vh_classification_core(ax, model.resolution, model.cells, alpha=0.8)
        elif plot_singular_cells:
            plot_cells_not_regular_classification_core(ax, model.resolution, model.cells, alpha=0.8)
        plot_curve_core(ax, curve_cells=[cell for cell in model.cells.values() if
                                         cell.CELL_TYPE != REGULAR_CELL_TYPE], color=winner_color_dict)

    draw_cell_borders(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )
    ax.set_xlim((-0.5 + trim[0][0], model.resolution[0] - trim[0][1] - 0.5))
    ax.set_ylim((model.resolution[1] - trim[1][0] - 0.5, trim[1][1] - 0.5))