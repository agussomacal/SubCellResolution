import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from experiments.Refinement.ex_refinement_config import experiment_path
from experiments.global_params import EVALUATIONS
from experiments.tools import calculate_averages_from_image, load_image, reconstruct, singular_cells_mask, \
    make_image_high_resolution, calculate_evaluations_from_curve, calculate_averages_from_curve
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.SubCellReconstruction import reconstruct_arbitrary_size, reconstruct_by_factor
from perplexitylab.experiment_tools import perplexifier


def single_experiment_vars_filter(*args, **kwargs):
    return list(map(str, args)) + list(map(str, kwargs.values()))


def image_to_avg(num_cells_per_dim, image, noise=0, seed=42):
    avg_values = calculate_averages_from_image(image, num_cells_per_dim)
    np.random.seed(seed)
    return avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)


def fit_model(sub_cell_model, angle_threshold, refinement, avg_values):
    model = sub_cell_model(refinement=refinement, angle_threshold=angle_threshold)

    t0 = time.time()
    model.fit(average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"))
    t_fit = time.time() - t0
    print("\n\tTime to fit model:", t_fit)
    return model


plx_fit_model = perplexifier(default_path=experiment_path)(fit_model)


@perplexifier(default_path=experiment_path,
              saver=lambda data, filepath: plt.imsave(filepath, data),
              loader=lambda filepath: load_image(filepath, other_path=""),
              file_format="png")
def do_reconstruction(image, model, reconstruction_factor, do_evaluations=EVALUATIONS):
    t0 = time.time()
    reconstruction = reconstruct(image, model.cells, model.resolution, reconstruction_factor,
                                 do_evaluations=do_evaluations)
    t_reconstruct = time.time() - t0
    print("\n\tTime to reconstruct:", t_reconstruct)
    return reconstruction


def calculate_error(image, reconstruction, p=2):
    return np.power(np.mean(np.abs(image - reconstruction) ** p), 1 / p)


def obtain_image4error(shape, num_cells_per_dim, sub_discretization2bound_error, avg_values, evaluation_mode=False):
    edge_mask = make_image_high_resolution(singular_cells_mask(avg_values),
                                           reconstruction_factor=sub_discretization2bound_error)
    cells2reconstruct = list(zip(*np.where(edge_mask)))
    true_reconstruction = make_image_high_resolution(avg_values, reconstruction_factor=sub_discretization2bound_error)

    if evaluation_mode:
        true_reconstruction[edge_mask] = calculate_evaluations_from_curve(
            shape, resolution=(num_cells_per_dim * sub_discretization2bound_error,
                               num_cells_per_dim * sub_discretization2bound_error),
            cells2reconstruct=cells2reconstruct)[edge_mask]
    else:
        true_reconstruction[edge_mask] = calculate_averages_from_curve(
            shape,
            (num_cells_per_dim * sub_discretization2bound_error,
             num_cells_per_dim * sub_discretization2bound_error),
            cells2reconstruct=cells2reconstruct)[edge_mask]

    return true_reconstruction


plx_obtain_image4error = perplexifier(default_path=experiment_path,
                                      saver=lambda data, filepath: plt.imsave(filepath, data),
                                      loader=lambda filepath: load_image(filepath, other_path=""),
                                      file_format="png")(obtain_image4error)


def efficient_reconstruction(model, avg_values, sub_discretization2bound_error, refinement, evaluation_mode=False,
                             threshold=1e-10):
    """
    Only reconstructs fully in the cells where there is discontinuity otherwise copies avgcells values
    :return:
    """

    edge_mask = singular_cells_mask(avg_values, threshold=threshold)
    edge_mask = np.repeat(np.repeat(edge_mask, 2 ** (refinement - 1), axis=0), 2 ** (refinement - 1), axis=1)
    cells2reconstruct = list(zip(*np.where(edge_mask)))

    reconstruction = np.repeat(np.repeat(avg_values, sub_discretization2bound_error, axis=0),
                               sub_discretization2bound_error, axis=1)

    magnification = sub_discretization2bound_error // 2 ** (refinement - 1)
    edge_mask_hr = np.repeat(np.repeat(edge_mask, magnification, axis=0), magnification, axis=1)
    reconstruction[edge_mask_hr] = \
        (
            reconstruct_arbitrary_size(cells=model.cells, resolution=model.resolution,
                                       cells2reconstruct=cells2reconstruct,
                                       size=np.shape(reconstruction))
            if evaluation_mode else
            reconstruct_by_factor(cells=model.cells, resolution=model.resolution, cells2reconstruct=cells2reconstruct,
                                  resolution_factor=magnification)

        )[edge_mask_hr]

    return reconstruction


plx_efficient_reconstruction = perplexifier(default_path=experiment_path,
                                            saver=lambda data, filepath: plt.imsave(filepath, data),
                                            loader=lambda filepath: load_image(filepath, other_path=""),
                                            file_format="png")(efficient_reconstruction)
