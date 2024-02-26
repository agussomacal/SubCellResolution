import time

import numpy as np

from PerplexityLab.visualization import perplex_plot, one_line_iterator
from experiments.VizReconstructionUtils import plot_cells, plot_cells_vh_classification_core, \
    plot_cells_not_regular_classification_core, plot_curve_core, draw_cell_borders, draw_numbers, \
    plot_cells_type_of_curve_core
from experiments.global_params import EVALUATIONS
from experiments.tools import load_image, calculate_averages_from_image, reconstruct, singular_cells_mask, \
    make_image_high_resolution
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import REGULAR_CELL_TYPE


def efficient_reconstruction(model, avg_values, sub_discretization2bound_error):
    """
    Only reconstructs fully in the cells where there is discontinuity otherwise copies avgcells values
    :return:
    """

    edge_mask = singular_cells_mask(avg_values)
    edge_mask_hr = make_image_high_resolution(edge_mask,
                                              reconstruction_factor=sub_discretization2bound_error)
    cells2reconstruct = list(zip(*np.where(edge_mask)))
    t0 = time.time()
    reconstruction = make_image_high_resolution(avg_values, reconstruction_factor=sub_discretization2bound_error)
    reconstruction[edge_mask_hr] = \
        reconstruct(image=avg_values, cells=model.cells, model_resolution=model.resolution,
                    cells2reconstruct=cells2reconstruct,
                    resolution_factor=sub_discretization2bound_error)[edge_mask_hr]
    t_reconstruct = time.time() - t0
    return reconstruction, t_reconstruct


def fit_model(sub_cell_model):
    def decorated_func(image, noise, num_cells_per_dim, reconstruction_factor, refinement, angle_threshold=25):
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


@perplex_plot(legend=False)
@one_line_iterator
def plot_reconstruction(fig, ax, image, num_cells_per_dim, model, reconstruction,
                        alpha=0.5, alpha_true_image=0.5, difference=False, plot_curve=True, plot_curve_winner=False,
                        plot_vh_classification=True, plot_singular_cells=True, cmap="viridis",
                        cmap_true_image="Greys_r", draw_mesh=True,
                        trim=((0, 1), (0, 1)),
                        numbers_on=True, vmin=None, vmax=None, labels=True, winner_color_dict=None):
    model_resolution = np.array(model.resolution)
    image = load_image(image)

    if alpha_true_image > 0:
        plot_cells(ax, colors=image, mesh_shape=model_resolution, alpha=alpha_true_image, cmap=cmap_true_image,
                   vmin=np.min(image) if vmin is None else vmin,
                   vmax=np.max(image) if vmax is None else vmax,
                   labels=labels)

    if difference:
        # TODO: should be the evaluations not the averages.
        image = calculate_averages_from_image(image, num_cells_per_dim=np.shape(reconstruction))
        d = reconstruction - image
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
            plot_cells_vh_classification_core(ax, model.resolution, model.cells, alpha=0.8)
        elif plot_singular_cells:
            plot_cells_not_regular_classification_core(ax, model.resolution, model.cells, alpha=0.8)
        plot_curve_core(ax, curve_cells=[cell for cell in model.cells.values() if
                                         cell.CELL_TYPE != REGULAR_CELL_TYPE], color=winner_color_dict)

    if draw_mesh:
        draw_cell_borders(
            ax, mesh_shape=num_cells_per_dim,
            refinement=model_resolution // num_cells_per_dim,
        )

    ax.set_ylim((model.resolution[1] - trim[0][1] - 0.5, -0.5 + trim[0][0]))
    ax.set_xlim((trim[1][0] - 0.5, model.resolution[0] - trim[1][1] - 0.5))

    draw_numbers(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )
