import operator
import time
from functools import partial

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import ClassPartialInit, NamedPartial
from PerplexityLab.visualization import generic_plot
from experiments.subcell_paper.function_families import load_image, calculate_averages_from_image
from experiments.subcell_paper.global_params import CCExtraWeight
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells, \
    get_opposite_regular_cells_by_stencil
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator
from lib.CellCreators.CurveCellCreators.VertexCellCreator import LinearVertexCellCurveCellCreator
from lib.CellCreators.RegularCellCreator import PiecewiseConstantRegularCellCreator
from lib.CellCreators.VertexCellCreators.VertexCellCreatorBase import VertexCellCreatorUsingNeighboursLines
from lib.CellIterators import iterate_by_condition_on_smoothness, iterate_by_reconstruction_error_and_smoothness, \
    iterate_all
from lib.CellOrientators import BaseOrientator, OrientByGradient, OrientPredefined
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.SmoothnessCalculators import naive_piece_wise
from lib.StencilCreators import StencilCreatorAdaptive, StencilCreatorFixedShape
from lib.SubCellReconstruction import SubCellReconstruction, ReconstructionErrorMeasure, CellCreatorPipeline, \
    reconstruct_arbitrary_size, reconstruct_by_factor
from lib.SubCellScheme import SubCellScheme

EVALUATIONS = True


def true_solution(image, num_cells_per_dim, velocity, ntimes):
    image = load_image(image)
    pixels_per_cell = np.array(np.shape(image)) / num_cells_per_dim
    velocity_in_pixels = np.array(pixels_per_cell * np.array(velocity), dtype=int)
    assert np.all(velocity_in_pixels == pixels_per_cell * np.array(velocity))

    true_solution = []
    for i in range(ntimes + 1):
        np.roll(image, velocity_in_pixels * i)
        true_solution.append(calculate_averages_from_image(image, num_cells_per_dim))

    return {"true_solution": true_solution}


def fit_model(subcell_reconstruction):
    def decorated_func(image, noise, num_cells_per_dim, reconstruction_factor, velocity, ntimes):
        image = load_image(image)
        avg_values = calculate_averages_from_image(image, num_cells_per_dim)
        np.random.seed(42)
        avg_values = avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)

        model = SubCellScheme(name=subcell_reconstruction.__name__, subcell_reconstructor=subcell_reconstruction())

        t0 = time.time()
        solution, all_cells = model.evolve(init_average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"),
                                           velocity=np.array(velocity), ntimes=ntimes)
        t_fit = time.time() - t0

        t0 = time.time()
        reconstruction = []
        for cells in all_cells:
            if EVALUATIONS:
                reconstruction.append(reconstruct_arbitrary_size(cells, model.resolution,
                                                                 np.array(np.shape(image)) // reconstruction_factor))
            else:
                reconstruction.append(reconstruct_by_factor(cells, model.resolution,
                                                            resolution_factor=np.array(
                                                                np.array(np.shape(image)) / np.array(model.resolution),
                                                                dtype=int)))
        t_reconstruct = time.time() - t0

        return {
            "model": model,
            "time_to_fit": t_fit,
            "reconstruction": reconstruction,
            "solution": solution,
            "time_to_reconstruct": t_reconstruct
        }

    # need to change the name so the lab experiment saves the correct name and not the uniformly "decorated_func"
    # the other option is to pass to the block the name we wish to associate to the function.
    decorated_func.__name__ = subcell_reconstruction.__name__
    return decorated_func


CurveAverageQuadraticCC = ClassPartialInit(CurveAveragePolynomial, class_name="CurveAverageQuadraticCC",
                                           degree=2, ccew=CCExtraWeight)


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
                cell_creator=PiecewiseConstantRegularCellCreator(
                    apriori_up_value=1, apriori_down_value=0, dimensionality=2)
            ),
        ],
        obera_iterations=0
    )


@fit_model
def elvira():
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
                cell_iterator=partial(iterate_by_condition_on_smoothness, value=REGULAR_CELL,
                                      condition=operator.eq),  # only regular cells
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
        ],
        obera_iterations=0
    )


@fit_model
def quadratic():
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
                cell_iterator=partial(iterate_by_condition_on_smoothness, value=REGULAR_CELL,
                                      condition=operator.eq),  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=PiecewiseConstantRegularCellCreator(
                    apriori_up_value=1, apriori_down_value=0, dimensionality=2)
            ),
            # ------------ AERO Quadratic ------------ #
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3),
                cell_creator=ValuesCurveCellCreator(
                    vander_curve=CurveAverageQuadraticCC,
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
        ],
        obera_iterations=0
    )


@fit_model
def full():
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
                cell_iterator=partial(iterate_by_condition_on_smoothness, value=REGULAR_CELL,
                                      condition=operator.eq),  # only regular cells
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
                orientator=OrientPredefined(predefined_axis=0),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3),
                cell_creator=ValuesCurveCellCreator(
                    vander_curve=CurveAverageQuadraticCC,
                    regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil)),
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=OrientPredefined(predefined_axis=1),
                stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3),
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
        true_solution,
        recalculate=False
    )

    lab.define_new_block_of_functions(
        "models",
        upwind,
        quadratic,
        elvira,
        recalculate=False
    )

    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=None,
        refinement=[1],
        ntimes=[10],
        velocity=[(0, 1 / 7)],
        num_cells_per_dim=[20],  # 20, 42, 84 168
        noise=[0],
        image=[
            "Elipsoid_1680x1680.jpg"
            # "yoda.jpg",
            # "DarthVader.jpeg",
            # "ShapesVertex_1680x1680.jpg",
            # "HandVertex_1680x1680.jpg",
            # "Polygon_1680x1680.jpg",
        ],
        iterations=[0],  # 500
        reconstruction_factor=[5],
    )

    scheme_error = lambda true_solution, solution: np.mean(
        np.abs((np.array(solution[1:]) - np.array(true_solution[1:]))), axis=(1, 2))
    times = lambda ntimes: np.arange(1, ntimes + 1)

    generic_plot(data_manager,
                 name="ErrorInTime",
                 x="times", y="scheme_error", label="models",
                 times=times, scheme_error=scheme_error,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="y",
                 )
