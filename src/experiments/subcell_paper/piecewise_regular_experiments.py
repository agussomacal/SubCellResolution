import time
from functools import partial

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.visualization import generic_plot
from experiments.image_reconstruction import plot_reconstruction
from experiments.subcell_paper.function_families import load_image, calculate_averages_from_image
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE
from lib.CellCreators.RegularCellCreator import MirrorCellCreator, \
    PolynomialRegularCellCreator, weight_cells_by_smoothness
from lib.CellCreators.WENOCellCreators import WENO16RegularCellCreator, WENO1DRegularCellCreator, \
    WENO1DPointsRegularCellCreator
from lib.CellIterators import iterate_all
from lib.CellOrientators import BaseOrientator
from lib.SmoothnessCalculators import indifferent, oracle
from lib.StencilCreators import StencilCreatorFixedShape, StencilCreatorSameRegionAdaptive
from lib.SubCellReconstruction import SubCellReconstruction, CellCreatorPipeline, ReconstructionErrorMeasureBase

import operator
import time
from functools import partial

import numpy as np

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import ClassPartialInit
from PerplexityLab.visualization import perplex_plot, one_line_iterator
from experiments.VizReconstructionUtils import plot_cells, plot_cells_identity, plot_cells_vh_classification_core, \
    plot_cells_not_regular_classification_core, plot_curve_core, draw_cell_borders
from experiments.subcell_paper.function_families import load_image, calculate_averages_from_image
from experiments.subcell_paper.global_params import CCExtraWeight, CurveAverageQuadraticCC
from experiments.subcell_paper.obera_experiments import get_sub_cell_model
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL, VERTEX_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE, REGULAR_CELL_TYPE, VERTEX_CELL_TYPE
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells, \
    get_opposite_regular_cells_by_stencil
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator
from lib.CellCreators.CurveCellCreators.VertexCellCreator import LinearVertexCellCurveCellCreator
from lib.CellCreators.RegularCellCreator import PiecewiseConstantRegularCellCreator
from lib.CellCreators.VertexCellCreators.VertexCellCreatorBase import VertexCellCreatorUsingNeighboursLines
from lib.CellIterators import iterate_by_condition_on_smoothness, iterate_by_reconstruction_error_and_smoothness
from lib.CellOrientators import BaseOrientator, OrientByGradient, OrientPredefined
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.SmoothnessCalculators import naive_piece_wise
from lib.StencilCreators import StencilCreatorAdaptive, StencilCreatorFixedShape
from lib.SubCellReconstruction import SubCellReconstruction, ReconstructionErrorMeasure, CellCreatorPipeline

EVALUATIONS = True


def enhance_image(image, amplitude):
    image = load_image(image)
    h, w = np.shape(image)
    y, x = np.meshgrid(*list(map(range, np.shape(image))))
    d = np.sqrt((x - h / 2) ** 2 + (y - w / 2) ** 2)
    # v=128+(64+32*sin((x-w/2+y-h/2)*5*6/w))*(v >0)-(v==0)*(64+32*cos(d*5*6/w))
    image += amplitude * (
            (image >= 0.5) * np.cos(2 * np.pi * x * 5 / w) +
            (image <= 0.5) * np.sin(2 * np.pi * d * 3 / w)
    )

    return {
        "enhanced_image": image
    }


def fit_model(sub_cell_model):
    def decorated_func(image, enhanced_image, noise, num_cells_per_dim, reconstruction_factor):
        image = load_image(image)
        not_perturbed_image = calculate_averages_from_image(image, num_cells_per_dim)
        avg_values = calculate_averages_from_image(enhanced_image, num_cells_per_dim)
        np.random.seed(42)
        avg_values = avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)

        model = sub_cell_model(
            partial(oracle, mask=(np.array(not_perturbed_image) > 0) * (np.array(not_perturbed_image) < 1)))

        t0 = time.time()
        model.fit(average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"))
        t_fit = time.time() - t0

        t0 = time.time()
        if EVALUATIONS:
            reconstruction = model.reconstruct_arbitrary_size(np.array(np.shape(image)) // reconstruction_factor)
        else:
            reconstruction = model.reconstruct_by_factor(
                resolution_factor=np.array(np.array(np.shape(image)) / np.array(model.resolution), dtype=int))
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


def get_reconstruction_error(enhanced_image, reconstruction, reconstruction_factor):
    if reconstruction_factor > 1:
        # TODO: should be the evaluations not the averages.
        enhanced_image = calculate_averages_from_image(enhanced_image, num_cells_per_dim=np.shape(reconstruction))
    return np.mean(np.abs(np.array(reconstruction) - enhanced_image))


reconstruction_error_measure = ReconstructionErrorMeasure(StencilCreatorFixedShape((3, 3)),
                                                          metric=2,
                                                          central_cell_extra_weight=1)
regular_deg2_same_region = CellCreatorPipeline(
    cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
                          condition=operator.eq),  # only regular cells
    orientator=BaseOrientator(dimensionality=2),
    stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=3),
    cell_creator=PolynomialRegularCellCreator(
        degree=2, noisy=False, weight_function=None,
        dimensionality=2, full_rank=True)
)
regular_deg1_same_region = CellCreatorPipeline(
    cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
                          condition=operator.eq),  # only regular cells
    orientator=BaseOrientator(dimensionality=2),
    stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=3),
    cell_creator=PolynomialRegularCellCreator(
        degree=1, noisy=False, weight_function=None,
        dimensionality=2, full_rank=True)
)
regular_constant_same_region = CellCreatorPipeline(
    cell_iterator=iterate_all,  # only regular cells
    orientator=BaseOrientator(dimensionality=2),
    stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
    cell_creator=MirrorCellCreator()
)

elvira_ccreator = CellCreatorPipeline(
    cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                          condition=operator.eq),
    orientator=OrientByGradient(kernel_size=(3, 3), dimensionality=2),
    stencil_creator=StencilCreatorFixedShape((3, 3)),
    cell_creator=ELVIRACurveCellCreator(
        regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil))


@fit_model
def poly2_elvira(smoothness_calculator):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_deg2_same_region,
            elvira_ccreator,
        ],
        obera_iterations=0
    )


@fit_model
def poly1_elvira(smoothness_calculator):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_deg1_same_region,
            elvira_ccreator,
        ],
        obera_iterations=0
    )


@fit_model
def poly0_elvira(smoothness_calculator):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_constant_same_region,
            elvira_ccreator,
        ],
        obera_iterations=0
    )


@fit_model
def quadratic(smoothness_calculator):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=ReconstructionErrorMeasure(StencilCreatorFixedShape((3, 3)),
                                                                metric=2,
                                                                central_cell_extra_weight=10),
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
                                      condition=operator.eq),  # only regular cells
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
def qelvira(smoothness_calculator):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=ReconstructionErrorMeasure(StencilCreatorFixedShape((3, 3)),
                                                                metric=2,
                                                                central_cell_extra_weight=1),
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
def full(smoothness_calculator):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=ReconstructionErrorMeasure(StencilCreatorFixedShape((3, 3)),
                                                                metric=2,
                                                                central_cell_extra_weight=100),
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
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


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='PieceWiseRegular',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()

    lab.define_new_block_of_functions(
        "perturbation",
        enhance_image
    )

    lab.define_new_block_of_functions(
        "models",
        poly0_elvira,
        poly1_elvira,
        poly2_elvira,
        recalculate=False
    )

    lab.execute(
        data_manager,
        num_cores=15,
        recalculate=False,
        forget=False,
        # amplitude=[0, 1e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1],
        amplitude=[1e-2],
        num_cells_per_dim=[20, 42, 84],  # 42 * 2
        noise=[0],
        image=[
            "Ellipsoid_1680x1680.jpg",
        ],
        reconstruction_factor=[6],
        # reconstruction_factor=[6],
    )

    generic_plot(data_manager, x="N", y="error", label="models",
                 # plot_func=NamedPartial(sns.lineplot, marker=".", linestyle="--"),
                 log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 error=get_reconstruction_error,
                 axes_by=["amplitude"],
                 plot_by=["reconstruction_factor"])

    # generic_plot(data_manager, x="amplitude", y="mse", label="models",
    #              # plot_func=NamedPartial(sns.lineplot, marker=".", linestyle="--"),
    #              log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
    #              mse=lambda reconstruction_error: np.mean(reconstruction_error),
    #              models=["weno16", "weno1d", "piecewise_constant"],
    #              axes_by=["num_cells_per_dim"],
    #              plot_by=["reconstruction_factor"])

    # generic_plot(data_manager, x="time", y="mse", label="models",
    #              # plot_func=NamedPartial(sns.lineplot, marker=".", linestyle="--"),
    #              log="xy", time=lambda time_to_fit: time_to_fit,
    #              mse=lambda reconstruction_error: np.mean(reconstruction_error),
    #              axes_by=["num_cells_per_dim"],
    #              plot_by=["reconstruction_factor"])

    plot_reconstruction(
        data_manager,
        name="Reconstruction",
        folder='reconstruction',
        axes_by=['models'],
        plot_by=["num_cells_per_dim", ],
        folder_by=['image', "amplitude", "reconstruction_factor"],
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
    # plot_original_image(
    #     data_manager,
    #     folder='reconstruction',
    #     axes_by=[],
    #     plot_by=['image', 'models', 'num_cells_per_dim', 'refinement'],
    #     axes_xy_proportions=(15, 15),
    #     numbers_on=True
    # )

    print("CO2 consumption: ", data_manager.CO2kg)
