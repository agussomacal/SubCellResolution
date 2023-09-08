import seaborn as sns

from PerplexityLab.visualization import generic_plot
from experiments.image_reconstruction import plot_reconstruction
from experiments.subcell_paper.tools import get_reconstruction_error, calculate_averages_from_image, load_image, \
    get_reconstruction_error_in_interface, reconstruct
from lib.CellCreators.RegularCellCreator import MirrorCellCreator, \
    PolynomialRegularCellCreator
from lib.CellIterators import iterate_all
from lib.SmoothnessCalculators import oracle
from lib.StencilCreators import StencilCreatorSameRegionAdaptive

import operator
import time
from functools import partial

import numpy as np

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from experiments.subcell_paper.global_params import CurveAverageQuadraticCC
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells, \
    get_opposite_regular_cells_by_stencil
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator
from lib.CellCreators.CurveCellCreators.VertexCellCreator import LinearVertexCellCurveCellCreator
from lib.CellCreators.RegularCellCreator import PiecewiseConstantRegularCellCreator
from lib.CellCreators.VertexCellCreators.VertexCellCreatorBase import VertexCellCreatorUsingNeighboursLines
from lib.CellIterators import iterate_by_reconstruction_error_and_smoothness
from lib.CellOrientators import BaseOrientator, OrientByGradient, OrientPredefined
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


reconstruction_error_measure = ReconstructionErrorMeasure(StencilCreatorFixedShape((3, 3)),
                                                          metric=2,
                                                          central_cell_extra_weight=100)
regular_deg2half_same_region = CellCreatorPipeline(
    cell_iterator=iterate_all,
    # cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
    #                       condition=operator.eq),  # only regular cells
    orientator=BaseOrientator(dimensionality=2),
    stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=3),
    cell_creator=PolynomialRegularCellCreator(
        degree=2, noisy=False, weight_function=None,
        dimensionality=2, full_rank=False)
)
regular_deg2_same_region = CellCreatorPipeline(
    cell_iterator=iterate_all,
    # cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
    #                       condition=operator.eq),  # only regular cells
    orientator=BaseOrientator(dimensionality=2),
    stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=3),
    cell_creator=PolynomialRegularCellCreator(
        degree=2, noisy=False, weight_function=None,
        dimensionality=2, full_rank=True)
)
regular_deg1_same_region = CellCreatorPipeline(
    cell_iterator=iterate_all,
    # cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
    #                       condition=operator.eq),  # only regular cells
    orientator=BaseOrientator(dimensionality=2),
    stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=3),
    cell_creator=PolynomialRegularCellCreator(
        degree=1, noisy=False, weight_function=None,
        dimensionality=2, full_rank=True)
)
regular_constant_same_region = CellCreatorPipeline(
    cell_iterator=iterate_all,
    # cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
    #                       condition=operator.eq),  # only regular cells
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

deg2cell_creator = [
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
            regular_opposite_cell_searcher=get_opposite_regular_cells_by_stencil))
]


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
def poly02_elvira(smoothness_calculator):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_constant_same_region,
            elvira_ccreator,
            regular_deg2_same_region,
        ],
        obera_iterations=0
    )


@fit_model
def poly2h_elvira(smoothness_calculator):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_deg2half_same_region,
            elvira_ccreator,

        ],
        obera_iterations=0
    )


@fit_model
def poly02h_elvira(smoothness_calculator):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_constant_same_region,
            elvira_ccreator,
            regular_deg2half_same_region,

        ],
        obera_iterations=0
    )


@fit_model
def poly02_qelvira(smoothness_calculator):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            [regular_constant_same_region, elvira_ccreator] +
            deg2cell_creator +
            [regular_deg2_same_region],
        ],
        obera_iterations=0
    )

    @fit_model
    def poly02h_qelvira(smoothness_calculator):
        return SubCellReconstruction(
            name="All",
            smoothness_calculator=smoothness_calculator,
            reconstruction_error_measure=reconstruction_error_measure,
            refinement=1,
            cell_creators=
            [  # regular cell with piecewise_constant
                [regular_constant_same_region, elvira_ccreator] +
                deg2cell_creator +
                [regular_deg2half_same_region],
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
            poly02_elvira,
            poly02_qelvira,
            poly2h_elvira,
            poly02h_elvira,
            poly02h_qelvira,
            recalculate=False
        )

        lab.execute(
            data_manager,
            num_cores=15,
            recalculate=False,
            save_on_iteration=15,
            forget=False,
            amplitude=[1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1],
            num_cells_per_dim=[20, 40],
            noise=[0],
            image=[
                "Ellipsoid_1680x1680.jpg",
            ],
            reconstruction_factor=[6],
            # reconstruction_factor=[6],
        )

        generic_plot(data_manager, x="amplitude", y="error", label="models",
                     plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                     log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     error=get_reconstruction_error,
                     plot_by=["reconstruction_factor", "N"])

        generic_plot(data_manager,
                     name="InterfaceError",
                     x="amplitude", y="interface_error", label="models",
                     plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                     log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     interface_error=get_reconstruction_error_in_interface,
                     plot_by=["reconstruction_factor", "N"])

        plot_reconstruction(
            data_manager,
            name="Reconstruction",
            folder='reconstruction',
            axes_by=["amplitude", ],
            plot_by=['models', ],
            folder_by=['image', "num_cells_per_dim", "reconstruction_factor"],
            axes_xy_proportions=(15, 15),
            # num_cells_per_dim=[20, 40],  # 42 * 2
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

        print("CO2 consumption: ", data_manager.CO2kg)
