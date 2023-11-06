import itertools
from typing import Union, Tuple

import numpy as np
from matplotlib import pylab as plt
from tqdm import tqdm

import config
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE
from lib.Curves.Curves import Curve, CurveBase
from lib.SubCellReconstruction import reconstruct_arbitrary_size, reconstruct_by_factor


def reconstruct(image: np.ndarray, cells, model_resolution, reconstruction_factor, do_evaluations=True):
    """

    :param do_evaluations: less time consuming.
    :return:
    """
    if do_evaluations:
        reconstruction = reconstruct_arbitrary_size(cells, model_resolution,
                                                    np.array(np.shape(image)) // reconstruction_factor)
    else:
        reconstruction = reconstruct_by_factor(cells, model_resolution,
                                               resolution_factor=np.array(
                                                   np.array(np.shape(image)) / np.array(model_resolution), dtype=int))
    return reconstruction


def calculate_averages_from_image(image, num_cells_per_dim: Union[int, Tuple[int, int]]):
    # Example of how to calculate the averages in a single pass:
    # np.arange(6 * 10).reshape((6, 10)).reshape((2, 3, 5, 2)).mean(-1).mean(-2)
    img_x, img_y = np.shape(image)
    ncx, ncy = (num_cells_per_dim, num_cells_per_dim) if isinstance(num_cells_per_dim, int) else num_cells_per_dim
    return image.reshape((ncx, img_x // ncx, ncy, img_y // ncy)).mean(-1).mean(-2)


def calculate_averages_from_curve(curve: Curve, resolution: Tuple[int, int], deplacement: Tuple = None,
                                  origin=(0, 0), cells2reconstruct=None):
    "TODO generalize"
    if deplacement is None:
        deplacement = 1 / np.array(resolution)
    assert deplacement[0] >= 0 and deplacement[1] >= 0, "only movements in positive implemented."

    num_squares = np.product(resolution)
    averages = np.zeros(resolution)
    for i, j in tqdm(
            itertools.product(*list(map(np.arange, resolution))) if cells2reconstruct is None else cells2reconstruct,
            desc="Over {}".format(num_squares)):
        averages[i, j] = curve.calculate_rectangle_average(
            x_limits=origin[0] + np.array(((i + 1) / resolution[0] - deplacement[0], (i + 1) / resolution[0])),
            y_limits=origin[1] + np.array(((j + 1) / resolution[1] - deplacement[1], (j + 1) / resolution[1]))
        )
    return averages * num_squares


def load_image(image_name):
    image = plt.imread(f"{config.images_path}/{image_name}", format=image_name.split(".")[-1])
    image = np.mean(image, axis=tuple(np.arange(2, len(np.shape(image)), dtype=int)))
    image -= np.min(image)
    image /= np.max(image)
    return image


def get_reconstruction_error(enhanced_image, reconstruction, reconstruction_factor):
    if reconstruction_factor > 1:
        # TODO: should be the evaluations not the averages.
        enhanced_image = calculate_averages_from_image(enhanced_image, num_cells_per_dim=np.shape(reconstruction))
    return np.mean(np.abs(np.array(reconstruction) - enhanced_image))


def singular_cells_mask(avg_values):
    return (0 < np.array(avg_values)) * (np.array(avg_values) < 1)


def make_image_high_resolution(matrix, reconstruction_factor):
    resolution_factor = reconstruction_factor if isinstance(reconstruction_factor, (list, tuple, np.ndarray)) else (
        reconstruction_factor, reconstruction_factor)
    return np.repeat(np.repeat(matrix, resolution_factor[0], axis=0), resolution_factor[1], axis=1)


def edge_mask(avg_values, reconstruction_factor):
    edge_mask = singular_cells_mask(avg_values)  # cells with an edge passing through
    # extend repeating mask to have the same shape as reconstructed images
    return make_image_high_resolution(edge_mask, reconstruction_factor)


def get_reconstruction_error_in_interface(image, enhanced_image, reconstruction, reconstruction_factor,
                                          num_cells_per_dim):
    mask = edge_mask(image, num_cells_per_dim, reconstruction)
    if reconstruction_factor > 1:
        # TODO: should be the evaluations not the averages.
        enhanced_image = calculate_averages_from_image(enhanced_image, num_cells_per_dim=np.shape(reconstruction))
    return np.mean(np.abs(np.array(reconstruction) - enhanced_image)[mask])


def curve_cells_fitting_times(model):
    return list(model.times[CURVE_CELL_TYPE].values())


def get_evaluations2test_curve(curve: CurveBase, kernel_size, refinement=5, center_cell_coords=None) -> np.ndarray:
    center_cell_coords = np.array(kernel_size) // 2 if center_cell_coords is None else center_cell_coords
    evaluations = np.array([curve(i - center_cell_coords[0] - 0.5, j - center_cell_coords[1] - 0.5) for i, j in
                            itertools.product(*list(map(lambda s: np.linspace(0, s, num=s * refinement, endpoint=False),
                                                        kernel_size)))]).reshape(
        tuple(np.array(kernel_size) * refinement))
    return evaluations
