import operator
import time
from functools import partial
from typing import Union, Tuple

import matplotlib.pylab as plt
import numpy as np

import config
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_cells_by_smoothness_threshold, \
    get_opposite_cells_by_grad, get_opposite_cells_by_relative_smoothness
from lib.CellCreators.RegularCellCreator import PolynomialRegularCellCreator, weight_cells
from lib.CellIterators import iterate_all, iterate_by_condition_on_smoothness, iterate_by_smoothness
from lib.CellOrientators import BaseOrientator, OrientPredefined, OrientByGradient
from lib.SmoothnessCalculators import indifferent, naive_piece_wise, by_gradient
from lib.StencilCreators import StencilCreatorSameRegionAdaptive, StencilCreatorFixedShape, \
    StencilCreatorSmoothnessDistTradeOff
from lib.SubCellReconstruction import CellCreatorPipeline, SubCellReconstruction, ReconstructionErrorMeasureBase, \
    ReconstructionErrorMeasure
from src.Indexers import ArrayIndexerNd


def calculate_averages_from_image(image, num_cells_per_dim: Union[int, Tuple[int, int]]):
    # Example of how to calculate the averages in a single pass:
    # np.arange(6 * 10).reshape((6, 10)).reshape((2, 3, 5, 2)).mean(-1).mean(-2)
    img_x, img_y = np.shape(image)
    ncx, ncy = (num_cells_per_dim, num_cells_per_dim) if isinstance(num_cells_per_dim, int) else num_cells_per_dim
    return image.reshape((ncx, img_x // ncx, ncy, img_y // ncy)).mean(-1).mean(-2)


def load_image(image_name):
    image = plt.imread(f"{config.images_path}/{image_name}", format=image_name.split(".")[-1])
    image = np.mean(image, axis=tuple(np.arange(2, len(np.shape(image)), dtype=int)))
    # image = plt.imread(f"{config.images_path}/{image_name}", format='jpeg').mean(-1)
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


def image_reconstruction(image, model):
    image = load_image(image)
    t0 = time.time()
    reconstruction = model.reconstruct_by_factor(
        resolution_factor=np.array(np.array(np.shape(image)) / np.array(model.resolution), dtype=int))
    t_reconstruct = time.time() - t0

    reconstruction_error = np.abs(np.array(reconstruction) - image)
    return {
        "reconstruction": reconstruction,
        "reconstruction_error": reconstruction_error,
        "time_to_reconstruct": t_reconstruct
    }


@fit_model_decorator
def piecewise_constant(refinement: int):
    return SubCellReconstruction(
        name="PiecewiseConstant",
        smoothness_calculator=indifferent,
        reconstruction_error_measure=ReconstructionErrorMeasureBase(),
        refinement=refinement,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=1, dimensionality=2),
                cell_creator=PolynomialRegularCellCreator(degree=0, dimensionality=2, noisy=False))
        ]
    )


@fit_model_decorator
def polynomial2(refinement: int):
    return SubCellReconstruction(
        name="Polynomial2",
        smoothness_calculator=by_gradient,
        reconstruction_error_measure=ReconstructionErrorMeasure(
            stencil_creator=StencilCreatorFixedShape((3, 3)), central_cell_extra_weight=1, metric="l2"),
        refinement=refinement,
        cell_creators=
        [  # regular cell with piecewise_constant
            # CellCreatorPipeline(
            #     cell_iterator=iterate_all,
            #     orientator=BaseOrientator(dimensionality=2),
            #     stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=1, dimensionality=2),
            #     cell_creator=PolynomialRegularCellCreator(degree=0, dimensionality=2, noisy=False)),
            # regular cell with quadratics
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # all cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorSmoothnessDistTradeOff(stencil_shape=(3, 3), dist_trade_off=0.5,
                                                                     avg_diff_trade_off=1),
                cell_creator=PolynomialRegularCellCreator(
                    degree=2, dimensionality=2, noisy=False, full_rank=False,
                    weight_function=partial(weight_cells, central_cell_extra_weight=100))
            )
        ]
    )


@fit_model_decorator
def elvira(refinement: int):
    return SubCellReconstruction(
        name="ELVIRA",
        smoothness_calculator=naive_piece_wise,
        reconstruction_error_measure=ReconstructionErrorMeasure(
            stencil_creator=StencilCreatorFixedShape((3, 3)), central_cell_extra_weight=1, metric="l2"),
        refinement=refinement,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_condition_on_smoothness, value=REGULAR_CELL,
                                      condition=operator.eq),  # only identified curve cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=1, dimensionality=2),
                cell_creator=PolynomialRegularCellCreator(dimensionality=2, noisy=False)
            )] +
        [  # curve cells with ELVIRA
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_condition_on_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=independent_axis, dimensionality=2),
                stencil_creator=StencilCreatorFixedShape((3, 3)),
                cell_creator=ELVIRACurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_cells_by_smoothness_threshold)
            ) for independent_axis in [0, 1]
        ]
    )


@fit_model_decorator
def elvira_soc(refinement: int):
    return SubCellReconstruction(
        name="ELVIRA",
        smoothness_calculator=by_gradient,
        reconstruction_error_measure=ReconstructionErrorMeasure(
            stencil_creator=StencilCreatorFixedShape((3, 3)), central_cell_extra_weight=1, metric="l2"),
        refinement=refinement,
        cell_creators=
        [   # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # all cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorSmoothnessDistTradeOff(stencil_shape=(3, 3), dist_trade_off=0.5,
                                                                     avg_diff_trade_off=1),
                cell_creator=PolynomialRegularCellCreator(
                    degree=2, dimensionality=2, noisy=False, full_rank=False,
                    weight_function=partial(weight_cells, central_cell_extra_weight=100))
            ),
            CellCreatorPipeline(
                cell_iterator=iterate_by_smoothness,
                orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
                stencil_creator=StencilCreatorFixedShape((3, 3)),
                cell_creator=ELVIRACurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_cells_by_grad)
            )
        ]
    )
