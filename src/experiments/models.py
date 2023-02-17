import time
from typing import Union, Tuple

import matplotlib.pylab as plt
import numpy as np

import config
from lib.CellCreators.RegularCellCreator import PolynomialRegularCellCreator
from lib.CellIterators import iterate_default
from lib.CellOrientators import BaseOrientator
from lib.SmoothnessCalculators import indifferent
from lib.StencilCreators import StencilCreatorSameRegionAdaptive
from lib.SubCellReconstruction import CellCreatorPipeline, SubCellReconstruction, ReconstructionErrorMeasureBase
from src.Indexers import ArrayIndexerNd


def calculate_averages_from_image(image, num_cells_per_dim: Union[int, Tuple[int, int]]):
    # Example of how to calculate the averages in a single pass:
    # np.arange(6 * 10).reshape((6, 10)).reshape((2, 3, 5, 2)).mean(-1).mean(-2)
    img_x, img_y = np.shape(image)
    ncx, ncy = (num_cells_per_dim, num_cells_per_dim) if isinstance(num_cells_per_dim, int) else num_cells_per_dim
    return image.reshape((ncx, img_x // ncx, ncy, img_y // ncy)).mean(-1).mean(-2)


def load_image(image_name):
    image = plt.imread(f"{config.images_path}/{image_name}", format='jpeg').mean(-1)
    image /= np.max(image)
    return image


def fit_model_decorator(function):
    def decorated_func(image, num_cells_per_dim, noise, refinement):
        image = load_image(image)
        np.random.seed(42)
        avg_values = calculate_averages_from_image(image, num_cells_per_dim)
        avg_values += np.random.uniform(-noise, noise, size=avg_values.shape)

        model = function(refinement)

        t0 = time.time()
        model.fit(average_values=avg_values,
                  indexer=ArrayIndexerNd(avg_values, "cyclic"))
        t_fit = time.time() - t0
        return {
            "model": model,
            "time_to_fit": t_fit
        }

    # need to change the name so the lab experiment saves the correct name and not the uniformly "decorated_func"
    # the other option is to pass to the block the name we wish to associate to the function.
    decorated_func.__name__ = function.__name__
    return decorated_func


@fit_model_decorator
def piecewise_constant(refinement: int):
    return SubCellReconstruction(
        name="PiecewiseConstant",
        smoothness_calculator=indifferent,
        reconstruction_error_measure=ReconstructionErrorMeasureBase(),
        refinement=refinement,
        cell_creators=[
            CellCreatorPipeline(
                cell_iterator=iterate_default,
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=1, dimensionality=2),
                cell_creator=PolynomialRegularCellCreator(dimensionality=2, noisy=False))
        ]
    )
