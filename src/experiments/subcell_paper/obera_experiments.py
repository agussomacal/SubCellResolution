import itertools
import operator
import time
from functools import partial

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline, FunctionBlock
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import perplex_plot, generic_plot, one_line_iterator
from experiments.VizReconstructionUtils import plot_cells, plot_cells_identity, plot_cells_vh_classification_core, \
    plot_cells_not_regular_classification_core, plot_curve_core, draw_cell_borders
from experiments.subcell_paper.function_families import load_image, calculate_averages_from_image, \
    calculate_averages_from_curve
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE, REGULAR_CELL_TYPE
from lib.CellCreators.CurveCellCreators.ParametersCurveCellCreators import DefaultCircleCurveCellCreator, \
    DefaultPolynomialCurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells
from lib.CellCreators.RegularCellCreator import PiecewiseConstantRegularCellCreator, MirrorCellCreator
from lib.CellIterators import iterate_by_condition_on_smoothness, iterate_all
from lib.CellOrientators import BaseOrientator, OrientByGradient
from lib.Curves.CurveCircle import CurveCircle, CircleParams, CurveSemiCircle
from lib.SmoothnessCalculators import naive_piece_wise, indifferent
from lib.StencilCreators import StencilCreatorFixedShape
from lib.SubCellReconstruction import SubCellReconstruction, CellCreatorPipeline, ReconstructionErrorMeasureBase, \
    ReconstructionErrorMeasure

EVALUATIONS = False  # if using evaluations or averages to measure error.


def get_sub_cell_model(curve_cell_creator, refinement, name, iterations, central_cell_extra_weight):
    return SubCellReconstruction(
        name=name,
        smoothness_calculator=naive_piece_wise,
        reconstruction_error_measure=ReconstructionErrorMeasure(StencilCreatorFixedShape((3, 3)),
                                                                metric="l2",
                                                                central_cell_extra_weight=central_cell_extra_weight),
        refinement=refinement,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_condition_on_smoothness, value=REGULAR_CELL,
                                      condition=operator.eq),  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=PiecewiseConstantRegularCellCreator(
                    apriori_up_value=1, apriori_down_value=0, dimensionality=2)
            ),
            # curve cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_condition_on_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
                stencil_creator=StencilCreatorFixedShape((3, 3)),
                cell_creator=curve_cell_creator(regular_opposite_cell_searcher=get_opposite_regular_cells))
        ],
        obera_iterations=iterations
    )


def fit_model(sub_cell_model):
    def decorated_func(image, noise, refinement, iterations, central_cell_extra_weight):
        # image = load_image(image)
        # avg_values = calculate_averages_from_image(image, num_cells_per_dim)
        np.random.seed(42)
        avg_values = image + np.random.uniform(-noise, noise, size=image.shape)

        model = sub_cell_model(refinement, iterations, central_cell_extra_weight)

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
    decorated_func.__name__ = sub_cell_model.__name__
    return decorated_func


@fit_model
def piecewise_constant(refinement: int, *args):
    return SubCellReconstruction(
        name="PiecewiseConstant",
        smoothness_calculator=indifferent,
        reconstruction_error_measure=ReconstructionErrorMeasureBase(),
        refinement=refinement,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=MirrorCellCreator(dimensionality=2)
            )
        ]
    )


# @fit_model
# def linear(refinement: int, iterations: int, central_cell_extra_weight: float):
#     return get_sub_cell_model(
#         partial(VanderCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=1)), refinement,
#         "Linear", iterations, central_cell_extra_weight)
#
#
# @fit_model
# def quadratic(refinement: int, iterations: int, central_cell_extra_weight: float):
#     return get_sub_cell_model(
#         partial(VanderCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=2)), refinement,
#         "Quadratic", iterations, central_cell_extra_weight)
#
#
# @fit_model
# def circle(refinement: int, iterations: int, central_cell_extra_weight: float):
#     return get_sub_cell_model(partial(VanderCurveCellCreator, vander_curve=CurveVanderCircle), refinement, "Circle",
#                               iterations, central_cell_extra_weight)


@fit_model
def linear(refinement: int, iterations: int, central_cell_extra_weight: float):
    return get_sub_cell_model(
        partial(DefaultPolynomialCurveCellCreator, degree=1), refinement,
        "Linear", iterations, central_cell_extra_weight)


@fit_model
def quadratic(refinement: int, iterations: int, central_cell_extra_weight: float):
    return get_sub_cell_model(
        partial(DefaultPolynomialCurveCellCreator, degree=2), refinement,
        "Quadratic", iterations, central_cell_extra_weight)


@fit_model
def circle(refinement: int, iterations: int, central_cell_extra_weight: float):
    return get_sub_cell_model(DefaultCircleCurveCellCreator, refinement, "Circle",
                              iterations, central_cell_extra_weight)


def image_reconstruction_from_curve(sub_discretization2bound_error, model):
    t0 = time.time()
    reconstruction = model.reconstruct_by_factor(resolution_factor=sub_discretization2bound_error)
    t_reconstruct = time.time() - t0
    return {
        "reconstruction": reconstruction,
        "time_to_reconstruct": t_reconstruct
    }


@perplex_plot()
@one_line_iterator
def plot_reconstruction(fig, ax, image4error, num_cells_per_dim, model, reconstruction, alpha=0.5,
                        plot_original_image=True,
                        difference=False, plot_curve=True, plot_curve_winner=False, plot_vh_classification=True,
                        plot_singular_cells=True, cmap="magma", trim=((0, 0), (0, 0)), numbers_on=True):
    model_resolution = np.array(model.resolution)

    if plot_original_image:
        plot_cells(ax, colors=image4error, mesh_shape=model_resolution, alpha=alpha, cmap="Greys_r",
                   vmin=np.min(image4error), vmax=np.max(image4error))
    if difference:
        plot_cells(ax, colors=reconstruction - image4error, mesh_shape=model_resolution, alpha=alpha, cmap=cmap,
                   vmin=-1, vmax=1)
    else:
        plot_cells(ax, colors=reconstruction, mesh_shape=model_resolution, alpha=alpha, cmap=cmap, vmin=-1, vmax=1)

    if plot_curve:
        if plot_curve_winner:
            plot_cells_identity(ax, model_resolution, model.cells, alpha=0.8)
            # plot_cells_type_of_curve_core(ax, model.resolution, model.cells, alpha=0.8)
        elif plot_vh_classification:
            plot_cells_vh_classification_core(ax, model_resolution, model.cells, alpha=0.8)
        elif plot_singular_cells:
            plot_cells_not_regular_classification_core(ax, model_resolution, model.cells, alpha=0.8)
        plot_curve_core(ax, curve_cells=[cell for cell in model.cells.values() if
                                         cell.CELL_TYPE == CURVE_CELL_TYPE])

    draw_cell_borders(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )
    ax.set_xlim((-0.5 + trim[0][0], model.resolution[0] - trim[0][1] - 0.5))
    ax.set_ylim((model.resolution[1] - trim[1][0] - 0.5, trim[1][1] - 0.5))


# def image_reconstruction(image, shape, model, reconstruction_factor):
#     # image = get_image(image)
#     t0 = time.time()
#     if EVALUATIONS:
#         reconstruction = model.reconstruct_arbitrary_size(np.array(np.shape(shape)) // reconstruction_factor)
#     else:
#         reconstruction = model.reconstruct_by_factor(
#             resolution_factor=np.array(np.array(np.shape(image)) / np.array(model.resolution), dtype=int))
#     t_reconstruct = time.time() - t0
#
#     if reconstruction_factor > 1:
#         if EVALUATIONS:
#             step = np.array(np.array(np.shape(image)) / np.array(np.shape(reconstruction)), dtype=int)
#             image = image[np.arange(0, np.shape(image)[0], step[0], dtype=int)][:,
#                     np.arange(0, np.shape(image)[1], step[1], dtype=int)]
#         else:
#             image = calculate_averages_from_image(image, num_cells_per_dim=np.shape(reconstruction))
#     reconstruction_error = np.abs(np.array(reconstruction) - image)
#     return {
#         "reconstruction": reconstruction,
#         "reconstruction_error": reconstruction_error,
#         "time_to_reconstruct": t_reconstruct
#     }


# @perplex_plot
# def plot_convergence_curves(fig, ax, num_cells_per_dim, reconstruction_error, models):
#     data = pd.DataFrame.from_dict({
#         "N": np.array(num_cells_per_dim) ** 2,
#         "Error": list(map(np.mean, reconstruction_error)),
#         "Model": models
#     })
#     for model, d in data.groupby("Model"):
#         plt.plot(d["N"], d["Error"], label=model)
#     ax.set_xticks(num_cells_per_dim, num_cells_per_dim)
#     y_ticks = np.arange(1 - int(np.log10(data["Error"].min())))
#     ax.set_yticks(10.0 ** (-y_ticks), [fr"$10^{-y}$" for y in y_ticks])
#     ax.set_yscale("log")
#     ax.set_xscale("log")
#     ax.legend()
#
#
# @perplex_plot()
# def plot_reconstruction(fig, ax, image, num_cells_per_dim, model, reconstruction, alpha=0.5,
#                         plot_original_image=True,
#                         difference=False, plot_curve=True, plot_curve_winner=False, plot_vh_classification=True,
#                         plot_singular_cells=True, cmap="magma", trim=((0, 0), (0, 0)), numbers_on=True, *args,
#                         **kwargs):
#     image = image.pop()
#     num_cells_per_dim = num_cells_per_dim.pop()
#     model = model.pop()
#     reconstruction = reconstruction.pop()
#
#     model_resolution = np.array(model.resolution)
#     image = load_image(image)
#
#     if plot_original_image:
#         plot_cells(ax, colors=image, mesh_shape=model_resolution, alpha=alpha, cmap="Greys_r",
#                    vmin=np.min(image), vmax=np.max(image))
#
#     if difference:
#         # TODO: should be the evaluations not the averages.
#         image = calculate_averages_from_image(image, num_cells_per_dim=np.shape(reconstruction))
#         plot_cells(ax, colors=reconstruction - image, mesh_shape=model.resolution, alpha=alpha, cmap=cmap, vmin=-1,
#                    vmax=1)
#     else:
#         plot_cells(ax, colors=reconstruction, mesh_shape=model.resolution, alpha=alpha, cmap=cmap, vmin=-1, vmax=1)
#
#     if plot_curve:
#         if plot_curve_winner:
#             plot_cells_identity(ax, model.resolution, model.cells, alpha=0.8)
#             # plot_cells_type_of_curve_core(ax, model.resolution, model.cells, alpha=0.8)
#         elif plot_vh_classification:
#             plot_cells_vh_classification_core(ax, model.resolution, model.cells, alpha=0.8)
#         elif plot_singular_cells:
#             plot_cells_not_regular_classification_core(ax, model.resolution, model.cells, alpha=0.8)
#         plot_curve_core(ax, curve_cells=[cell for cell in model.cells.values() if
#                                          cell.CELL_TYPE == CURVE_CELL_TYPE])
#
#     draw_cell_borders(
#         ax, mesh_shape=num_cells_per_dim,
#         refinement=model_resolution // num_cells_per_dim,
#         numbers_on=numbers_on,
#         prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
#     )
#     ax.set_xlim((-0.5 + trim[0][0], model.resolution[0] - trim[0][1] - 0.5))
#     ax.set_ylim((model.resolution[1] - trim[1][0] - 0.5, trim[1][1] - 0.5))


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='OBERA',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )


    def get_shape(shape_name):
        if shape_name == "Circle":
            return CurveCircle(params=CircleParams(x0=0.511, y0=0.486, radius=0.232))
            # # - CurveCircle(params=CircleParams(x0=0.511, y0=0.486, radius=0.232), concave=True)
            # return CurveSemiCircle(params=CircleParams(x0=0.511, y0=0.486, radius=0.232)) \
            #        - CurveSemiCircle(params=CircleParams(x0=0.511, y0=0.486, radius=0.232), concave=True)
        # SFSemiCircle(x0=x0, y0=y0, radius=radius) - SFSemiCircle(x0=x0, y0=y0, radius=radius, concave=True)
        else:
            raise Exception("Not implemented.")


    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "precompute_images",
        FunctionBlock(
            "getimages",
            lambda shape_name, num_cells_per_dim: {
                "image": calculate_averages_from_curve(
                    get_shape(shape_name),
                    (num_cells_per_dim,
                     num_cells_per_dim))}
        )
    )

    lab.define_new_block_of_functions(
        "precompute_error_resolution",
        FunctionBlock(
            "subresolution",
            lambda shape_name, num_cells_per_dim, sub_discretization2bound_error: {
                "image4error": calculate_averages_from_curve(
                    get_shape(shape_name),
                    (num_cells_per_dim * sub_discretization2bound_error,
                     num_cells_per_dim * sub_discretization2bound_error))}
        )
    )

    lab.define_new_block_of_functions(
        "models",
        # piecewise_constant,
        linear,
        quadratic,
        circle
    )

    lab.define_new_block_of_functions(
        "image_reconstruction",
        image_reconstruction_from_curve
    )

    lab.execute(
        data_manager,
        num_cores=15,
        recalculate=False,
        forget=False,
        save_on_iteration=1,
        refinement=[1],
        num_cells_per_dim=[10, 14] + np.logspace(np.log10(20), np.log10(100), num=10, dtype=int).tolist()[:4],
        # num_cells_per_dim=[14],
        # num_cells_per_dim=[8, 14, 20, 28, 42, 42 * 2],  # 42 * 2
        noise=[0],
        shape_name=[
            # "Ellipsoid_1680x1680.png",
            "Circle"
        ],
        iterations=[500],  # 500
        # reconstruction_factor=[1],
        # central_cell_extra_weight=[0],
        central_cell_extra_weight=[0, 100],
        sub_discretization2bound_error=[5]
    )

    generic_plot(data_manager, x="N", y="time", label="models",
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="x", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 time=lambda model: np.array(list(model.times[CURVE_CELL_TYPE].values())),
                 # error=lambda reconstruction, image4error: np.mean(np.abs(np.array(reconstruction) - image4error)),
                 ylim=(0, 0.8),
                 plot_by=["iterations", "central_cell_extra_weight"])

    generic_plot(data_manager, x="N", y="fevals", label="models",
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="x", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 fevals=lambda model: np.array(list(model.obera_fevals[CURVE_CELL_TYPE].values())),
                 # error=lambda reconstruction, image4error: np.mean(np.abs(np.array(reconstruction) - image4error)),
                 ylim=(0, 225),
                 plot_by=["iterations", "central_cell_extra_weight"])


    def get_reconstructed_subcells_coords(coord, sub_discretization2bound_error, reconstruction):
        return reconstruction[list(
            map(lambda i: np.arange(i * sub_discretization2bound_error, (i + 1) * sub_discretization2bound_error),
                coord))]


    def singular_error(reconstruction, image4error, model, sub_discretization2bound_error, num_cells_per_dim):
        return np.array(list(map(np.mean,
                                 map(partial(get_reconstructed_subcells_coords,
                                             reconstruction=np.abs(np.array(reconstruction) - image4error),
                                             sub_discretization2bound_error=sub_discretization2bound_error),
                                     model.obera_fevals[CURVE_CELL_TYPE].keys()
                                     )
                                 )
                             ))/num_cells_per_dim**2


    generic_plot(data_manager, x="fevals", y="error", label="models",
                 plot_func=NamedPartial(sns.scatterplot, marker="o"),
                 log="xy",
                 # N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 time=lambda model: np.array(list(model.times[CURVE_CELL_TYPE].values())),
                 fevals=lambda model: np.array(list(model.obera_fevals[CURVE_CELL_TYPE].values())),
                 error=singular_error,
                 ylim=(1e-13, 1e-2),
                 plot_by=["iterations"])

    # generic_plot(data_manager, x="N", y="error", label="models",
    #              plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
    #              log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
    #              error=lambda reconstruction, image4error: np.mean(np.abs(np.array(reconstruction) - image4error)),
    #              plot_by=["iterations", "central_cell_extra_weight"])
    #
    # generic_plot(data_manager, x="time", y="error", label="models",
    #              plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
    #              log="xy", time=lambda time_to_fit: time_to_fit,
    #              error=lambda reconstruction, image4error: np.mean(np.abs(np.array(reconstruction) - image4error)),
    #              plot_by=["iterations", "central_cell_extra_weight"])
    #
    # generic_plot(data_manager, x="N", y="time", label="models",
    #              plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
    #              log="xy", time=lambda time_to_fit: time_to_fit,
    #              N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
    #              plot_by=["iterations", "central_cell_extra_weight"])

    # def get_number_of_curve_cells(model):
    #     return sum([1 for cell in model.cells.values() if cell.CELL_TYPE == CURVE_CELL_TYPE])
    #
    #
    # generic_plot(data_manager, y="time", x="iterations", label="models",
    #              plot_func=sns.barplot,
    #              models=["linear", "quadratic", "circle"],
    #              time=lambda time_to_fit, model: time_to_fit / get_number_of_curve_cells(model))

    plot_reconstruction(
        data_manager,
        name="Reconstruction",
        folder='reconstruction',
        num_cells_per_dim=14,
        axes_by=['models'],
        # plot_by=['shape_name', "num_cells_per_dim", 'refinement', "iterations", 'central_cell_extra_weight'],
        plot_by=['central_cell_extra_weight'],
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
        trim=((3, 7), (3, 6))
    )

    # plot_original_image(
    #     data_manager,
    #     folder='reconstruction',
    #     axes_by=[],
    #     plot_by=['image', 'models', 'num_cells_per_dim', 'refinement'],
    #     axes_xy_proportions=(15, 15),
    #     numbers_on=True
    # )

    print("CO2 consumption: ", data_manager.CO2kg)
