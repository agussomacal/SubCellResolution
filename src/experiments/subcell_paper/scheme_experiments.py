import operator
import time
from functools import partial

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot, one_line_iterator, perplex_plot
from experiments.VizReconstructionUtils import plot_cells, draw_cell_borders, plot_cells_identity, \
    plot_cells_vh_classification_core, plot_cells_not_regular_classification_core, plot_curve_core
from experiments.subcell_paper.global_params import CurveAverageQuadraticCC
from experiments.subcell_paper.tools import get_reconstruction_error, calculate_averages_from_image, load_image, \
    reconstruct
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import REGULAR_CELL_TYPE
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells, \
    get_opposite_regular_cells_by_stencil
from lib.CellCreators.CurveCellCreators.TaylorCurveCellCreator import TaylorCircleCurveCellCreator
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator
from lib.CellCreators.CurveCellCreators.VertexCellCreator import LinearVertexCellCurveCellCreator
from lib.CellCreators.RegularCellCreator import PiecewiseConstantRegularCellCreator, MirrorCellCreator
from lib.CellCreators.VertexCellCreators.VertexCellCreatorBase import VertexCellCreatorUsingNeighboursLines
from lib.CellIterators import iterate_by_reconstruction_error_and_smoothness, \
    iterate_all
from lib.CellOrientators import BaseOrientator, OrientByGradient, OrientPredefined
from lib.SmoothnessCalculators import naive_piece_wise
from lib.StencilCreators import StencilCreatorAdaptive, StencilCreatorFixedShape
from lib.SubCellReconstruction import SubCellReconstruction, ReconstructionErrorMeasure, CellCreatorPipeline
from lib.SubCellScheme import SubCellScheme

EVALUATIONS = True


# N = int(1e6)
# workers = 10
# dataset_manager_3_8pi = DatasetsManagerLinearCurves(
#     velocity_range=((0, 0), (0, 1)), path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
#     workers=workers, recalculate=False, learning_objective=ANGLE_OBJECTIVE, angle_limits=(-3 / 8, 3 / 8),
#     value_up_random=False
# )

def calculate_true_solution(image, num_cells_per_dim, velocity, ntimes):
    image = load_image(image)
    pixels_per_cell = np.array(np.shape(image)) / num_cells_per_dim
    velocity_in_pixels = np.array(pixels_per_cell * np.array(velocity), dtype=int)
    assert np.all(velocity_in_pixels == pixels_per_cell * np.array(velocity))

    true_solution = []
    true_reconstruction = []
    for i in range(ntimes + 1):
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

        t0 = time.time()
        solution, all_cells = model.evolve(
            init_average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"),
            velocity=np.array(velocity), ntimes=ntimes,
            # interface_oracle=None
            interface_oracle=(np.array(true_solution) > 0) * (np.array(true_solution) < 1)
        )
        t_fit = time.time() - t0

        t0 = time.time()
        reconstruction = []
        for cells in all_cells:
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


reconstruction_error_measure = ReconstructionErrorMeasure(StencilCreatorFixedShape((3, 3)),
                                                          metric=2, central_cell_extra_weight=1)


@fit_model
def upwind():
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=naive_piece_wise,
        reconstruction_error_measure=ReconstructionErrorMeasure(StencilCreatorFixedShape((3, 3)),
                                                                metric=2,
                                                                central_cell_extra_weight=100),
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=MirrorCellCreator(dimensionality=2)
            ),
        ],
        obera_iterations=0
    )


@fit_model
def elvira():
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=naive_piece_wise,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=PiecewiseConstantRegularCellCreator(
                    apriori_up_value=1, apriori_down_value=0, dimensionality=2)
            ),
            # CellCreatorPipeline(
            #     cell_iterator=iterate_all,  # only regular cells
            #     orientator=BaseOrientator(dimensionality=2),
            #     stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
            #     cell_creator=MirrorCellCreator(dimensionality=2)
            # ),
            # ------------ ELVIRA ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
                stencil_creator=StencilCreatorFixedShape((3, 3)),
                cell_creator=ELVIRACurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
        ],
        obera_iterations=0
    )


@fit_model
def quadratic():
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=partial(naive_piece_wise, eps=1e-2, min_val=0, max_val=1),
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=PiecewiseConstantRegularCellCreator(
                    apriori_up_value=1, apriori_down_value=0, dimensionality=2)
            ),
            # CellCreatorPipeline(
            #     cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
            #                           condition=operator.eq),  # only regular cells
            #     orientator=BaseOrientator(dimensionality=2),
            #     stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
            #     cell_creator=MirrorCellCreator(dimensionality=2)
            # ),
            # ------------ ELVIRA ------------ #
            # CellCreatorPipeline(
            #     cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
            #                           condition=operator.eq),
            #     orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
            #     stencil_creator=StencilCreatorFixedShape((3, 3)),
            #     cell_creator=ELVIRACurveCellCreator(
            #         regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            # ------------ AERO Quadratic ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=0, dimensionality=2),
                # orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                       independent_dim_stencil_size=3),
                cell_creator=ValuesCurveCellCreator(
                    vander_curve=CurveAverageQuadraticCC,
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=1, dimensionality=2),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                       independent_dim_stencil_size=3),
                cell_creator=ValuesCurveCellCreator(
                    vander_curve=CurveAverageQuadraticCC,
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
        ],
        obera_iterations=0
    )


@fit_model
def qelvira():
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=naive_piece_wise,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=PiecewiseConstantRegularCellCreator(
                    apriori_up_value=1, apriori_down_value=0, dimensionality=2)
            ),
            # CellCreatorPipeline(
            #     cell_iterator=iterate_all,  # only regular cells
            #     orientator=BaseOrientator(dimensionality=2),
            #     stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
            #     cell_creator=MirrorCellCreator(dimensionality=2)
            # ),
            # ------------ ELVIRA ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
                stencil_creator=StencilCreatorFixedShape((3, 3)),
                cell_creator=ELVIRACurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            # ------------ AERO Quadratic ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=0, dimensionality=2),
                # orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                       independent_dim_stencil_size=3),
                cell_creator=ValuesCurveCellCreator(
                    vander_curve=CurveAverageQuadraticCC,
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=1, dimensionality=2),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                       independent_dim_stencil_size=3),
                cell_creator=ValuesCurveCellCreator(
                    vander_curve=CurveAverageQuadraticCC,
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
        ],
        obera_iterations=0
    )


@fit_model
def elviracircle():
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=partial(naive_piece_wise, eps=1e-2, min_val=0, max_val=1),
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=PiecewiseConstantRegularCellCreator(
                    apriori_up_value=1, apriori_down_value=0, dimensionality=2)
            ),
            # ------------ ELVIRA ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
                stencil_creator=StencilCreatorFixedShape((3, 3)),
                cell_creator=ELVIRACurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            # CellCreatorPipeline(
            #     cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
            #                           condition=operator.eq),  # only regular cells
            #     orientator=BaseOrientator(dimensionality=2),
            #     stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
            #     cell_creator=MirrorCellCreator(dimensionality=2)
            # ),
            # ------------ ELVIRA ------------ #
            # CellCreatorPipeline(
            #     cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
            #                           condition=operator.eq),
            #     orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
            #     stencil_creator=StencilCreatorFixedShape((3, 3)),
            #     cell_creator=ELVIRACurveCellCreator(
            #         regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            # ------------ AERO Circle ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                # orientator=OrientPredefined(predefined_axis=0, dimensionality=2),
                orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                       independent_dim_stencil_size=3, center_weight=2.1),
                cell_creator=TaylorCircleCurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=1, dimensionality=2),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                       independent_dim_stencil_size=3, center_weight=2.1),
                cell_creator=TaylorCircleCurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
        ],
        obera_iterations=0
    )


@fit_model
def full():
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=naive_piece_wise,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=PiecewiseConstantRegularCellCreator(
                    apriori_up_value=1, apriori_down_value=0, dimensionality=2)
            ),
            # ------------ ELVIRA ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
                stencil_creator=StencilCreatorFixedShape((3, 3)),
                cell_creator=ELVIRACurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            # ------------ TEM ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=0),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 3)),
                cell_creator=VertexCellCreatorUsingNeighboursLines(
                    regular_opposite_cell_searcher=partial(get_opposite_regular_cells, direction="grad"))
            ),
            # ------------ AERO Quadratic ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=0, dimensionality=2),
                # orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                       independent_dim_stencil_size=3),
                cell_creator=ValuesCurveCellCreator(
                    vander_curve=CurveAverageQuadraticCC,
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=1, dimensionality=2),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                       independent_dim_stencil_size=3),
                cell_creator=ValuesCurveCellCreator(
                    vander_curve=CurveAverageQuadraticCC,
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            # ------------ AVRO ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=0),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=4),
                cell_creator=LinearVertexCellCurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells)
            ),
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=1),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=4),
                cell_creator=LinearVertexCellCurveCellCreator(
                    regular_opposite_cell_searcher=get_opposite_regular_cells)
            ),
        ],
        obera_iterations=0
    )


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
        # upwind,
        quadratic,
        # elvira,
        # qelvira,
        full,
        recalculate=True
    )

    lab.execute(
        data_manager,
        num_cores=1,
        forget=False,
        save_on_iteration=5,
        refinement=[1],
        ntimes=[10],
        velocity=[(0, 1 / 4)],
        num_cells_per_dim=[20],  # 60
        noise=[0],
        image=[
            "Ellipsoid_1680x1680.jpg",
            # "yoda.jpg",
            # "DarthVader.jpeg",
            "ShapesVertex_1680x1680.jpg",
            "HandVertex_1680x1680.jpg",
            "Polygon_1680x1680.jpg",
        ],
        iterations=[0],  # 500
        reconstruction_factor=[5],
    )

    scheme_error = lambda image, true_solution, solution: np.mean(
        np.abs((np.array(solution[1:]) - np.array(true_solution[1:]))), axis=(1, 2))

    scheme_reconstruction_error = lambda true_reconstruction, reconstruction, reconstruction_factor: np.array([
        get_reconstruction_error(tr_i, reconstruction=r_i, reconstruction_factor=reconstruction_factor)
        for tr_i, r_i in zip(true_reconstruction[:-1], reconstruction)])

    generic_plot(data_manager,
                 name="ErrorInTime",
                 x="times", y="scheme_error", label="models", plot_by=["num_cells_per_dim", "image"],
                 # models=["elvira", "quadratic"],
                 times=lambda ntimes: np.arange(1, ntimes + 1), scheme_error=scheme_error,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="y",
                 )

    generic_plot(data_manager,
                 name="ReconstructionErrorInTime",
                 x="times", y="scheme_error", label="models", plot_by=["num_cells_per_dim", "image"],
                 # models=["elvira", "quadratic"],
                 times=lambda ntimes: np.arange(ntimes), scheme_error=scheme_reconstruction_error,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="y",
                 )

    for i in range(10):
        plot_time_i(data_manager, folder="Solution", name=f"Time{i}", i=i, alpha=0.8, cmap="viridis",
                    trim=((0, 0), (0, 0)), folder_by=['image', 'num_cells_per_dim'],
                    plot_by=[],
                    axes_by=["models"],
                    numbers_on=True, error=True)

        plot_reconstruction_time_i(
            data_manager,
            i=i,
            name=f"Reconstruction{i}",
            folder='Reconstruction',
            folder_by=['image', 'num_cells_per_dim'],
            plot_by=[],
            axes_by=["models"],
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
