import operator
import time
from functools import partial

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot
from experiments.OtherExperiments.SubcellExperiments.ex_regular import trigonometric
from experiments.global_params import CCExtraWeight, EVALUATIONS, cpink, corange, cred, cgreen, cblue
from experiments.PaperPlots.models2compare import reconstruction_error_measure_3x3_w
from experiments.tools import get_reconstruction_error, calculate_averages_from_image, load_image, \
    reconstruct
from experiments.tools4binary_images import plot_reconstruction
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CurveCellCreators.CurveCellCreatorBase import get_values_up_down_eval, get_values_up_down_avg, \
    get_values_up_down_regcell_eval, get_values_up_down_regcell_avg
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells_by_minmax
from lib.CellCreators.RegularCellCreator import PolynomialRegularCellCreator, weight_cells_by_distance
from lib.CellIterators import iterate_all, iterate_by_reconstruction_error_and_smoothness
from lib.CellOrientators import BaseOrientator, OrientByGradient
from lib.SmoothnessCalculators import oracle
from lib.StencilCreators import StencilCreatorFixedShape
from lib.StencilCreators import StencilCreatorSameRegionAdaptive
from lib.SubCellReconstruction import SubCellReconstruction, ReconstructionErrorMeasure, CellCreatorPipeline

ELVIRA_CC_WEIGHT = 0
ANGLE_THRESHOLD = 30

updown_value_getter_dict = {
    "SCell eval": get_values_up_down_eval,
    "SCell evg": get_values_up_down_avg,
    "RCell eval": get_values_up_down_regcell_eval,
    "RCell avg": get_values_up_down_regcell_avg,

}

updown_value_color_dict = {
    "SCell eval": cblue,
    "SCell evg": cgreen,
    "RCell eval": cred,
    "RCell avg": corange,

}


def fit_model(sub_cell_model):
    def decorated_func(image, enhanced_image, noise, num_cells_per_dim, reconstruction_factor, updown_method):
        image = load_image(image)
        not_perturbed_image = calculate_averages_from_image(image, num_cells_per_dim)
        avg_values = calculate_averages_from_image(enhanced_image, num_cells_per_dim)
        np.random.seed(42)
        avg_values = avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)

        model = sub_cell_model(
            smoothness_calculator=partial(oracle,
                                          mask=(np.array(not_perturbed_image) > 0) * (
                                                  np.array(not_perturbed_image) < 1)),
            updown_value_getter=updown_value_getter_dict[updown_method]
        )

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


def elvira_cc(angle_threshold, weight=CCExtraWeight, updown_value_getter=get_values_up_down_regcell_eval):
    return CellCreatorPipeline(
        cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=CURVE_CELL,
                              condition=operator.eq),
        orientator=OrientByGradient(kernel_size=(5, 5), dimensionality=2, method="optim",
                                    angle_threshold=angle_threshold),
        stencil_creator=StencilCreatorFixedShape((3, 3)),
        cell_creator=ELVIRACurveCellCreator(
            regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax,
            updown_value_getter=updown_value_getter
        ),
        reconstruction_error_measure=reconstruction_error_measure_3x3_w(weight)
    )


reconstruction_error_measure = ReconstructionErrorMeasure(StencilCreatorFixedShape((3, 3)),
                                                          metric=2,
                                                          central_cell_extra_weight=100)
regular_deg2half_same_region = CellCreatorPipeline(
    # cell_iterator=iterate_all,
    cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
                          condition=operator.eq),  # only regular cells
    orientator=BaseOrientator(dimensionality=2),
    stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=3),
    # stencil_creator=StencilCreatorFixedShape(stencil_shape=(5, 5)),
    cell_creator=PolynomialRegularCellCreator(
        degree=2, noisy=False,
        weight_function=partial(weight_cells_by_distance, central_cell_importance=CCExtraWeight, distance_weight=0.5),
        # weight_function=partial(weight_cells_by_smoothness, central_cell_importance=CCExtraWeight, epsilon=1e-5,
        #                         delta=0.05),
        dimensionality=2, full_rank=False)
)
regular_deg2_same_region = CellCreatorPipeline(
    # cell_iterator=iterate_all,
    cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
                          condition=operator.eq),  # only regular cells
    orientator=BaseOrientator(dimensionality=2),
    stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=3),
    # cell_creator=PolynomialRegularCellCreator(
    #     degree=2, noisy=False, weight_function=None,
    #     dimensionality=2, full_rank=True)
    # stencil_creator=StencilCreatorFixedShape(stencil_shape=(5, 5)),
    cell_creator=PolynomialRegularCellCreator(
        degree=2, noisy=False,
        weight_function=partial(weight_cells_by_distance, central_cell_importance=CCExtraWeight, distance_weight=0.5),
        dimensionality=2, full_rank=True)
)
regular_deg1_same_region = CellCreatorPipeline(
    # cell_iterator=iterate_all,
    cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
                          condition=operator.eq),  # only regular cells
    orientator=BaseOrientator(dimensionality=2),
    stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=3),
    cell_creator=PolynomialRegularCellCreator(
        degree=1, noisy=False, weight_function=None,
        dimensionality=2, full_rank=True)
)

regular_constant_same_region = CellCreatorPipeline(
    # cell_iterator=iterate_all,
    cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
                          condition=operator.eq),  # only regular cells
    orientator=BaseOrientator(dimensionality=2),
    stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=1),
    cell_creator=PolynomialRegularCellCreator(degree=0)
)


# regular_constant_same_region = CellCreatorPipeline(
#     cell_iterator=iterate_all,
#     # cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
#     #                       condition=operator.eq),  # only regular cells
#     orientator=BaseOrientator(dimensionality=2),
#     stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
#     cell_creator=MirrorCellCreator()
# )

@fit_model
def poly02half(smoothness_calculator):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=3),
                cell_creator=PolynomialRegularCellCreator(degree=0)
            ),
            CellCreatorPipeline(
                cell_iterator=iterate_all,
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=3),
                cell_creator=PolynomialRegularCellCreator(
                    degree=2, noisy=False,
                    weight_function=partial(weight_cells_by_distance, central_cell_importance=CCExtraWeight,
                                            distance_weight=0.5),
                    dimensionality=2, full_rank=False)
            ),
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
            elvira_cc(angle_threshold=ANGLE_THRESHOLD, weight=ELVIRA_CC_WEIGHT),
        ],
        obera_iterations=0
    )


@fit_model
def poly2_elvira(smoothness_calculator, updown_value_getter):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_deg2_same_region,
            elvira_cc(angle_threshold=ANGLE_THRESHOLD, weight=ELVIRA_CC_WEIGHT,
                      updown_value_getter=updown_value_getter),
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
            elvira_cc(angle_threshold=ANGLE_THRESHOLD, weight=ELVIRA_CC_WEIGHT),
            regular_deg2_same_region,
        ],
        obera_iterations=0
    )


@fit_model
def poly2h_elvira(smoothness_calculator, updown_value_getter):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_deg2half_same_region,
            elvira_cc(angle_threshold=ANGLE_THRESHOLD, weight=ELVIRA_CC_WEIGHT,
                      updown_value_getter=updown_value_getter),
        ],
        obera_iterations=0
    )


if __name__ == "__main__":
    name = 'ValueUpDown'

    models = [
        # poly02half,
        # poly0_elvira,
        poly2_elvira,
        poly2h_elvira,
        # poly02_elvira,
    ]

    data_manager = DataManager(
        path=config.results_path,
        name=name,
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()

    lab.define_new_block_of_functions(
        "perturbation",
        trigonometric,
        # parabolas
    )

    lab.define_new_block_of_functions(
        "models",
        *models,
        recalculate=False
    )

    lab.execute(
        data_manager,
        num_cores=15,
        recalculate=False,
        save_on_iteration=None,
        forget=False,
        frequency=[0.5, 2],
        # frequency=[2],
        amplitude=[1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1],
        # amplitude=[1e-4],
        num_cells_per_dim=[20, 40],
        # num_cells_per_dim=[40],
        noise=[0],
        image=[
            "Ellipsoid_1680x1680.jpg",
        ],
        reconstruction_factor=[5],
        updown_method=list(updown_value_getter_dict.keys())
    )

    names_dict = {
        "poly02half": "Constant + Quadratic",
        "poly0_elvira": "Constant + ELVIRA",
        "poly2_elvira": "Quadratic full + ELVIRA",
        "poly2h_elvira": "Quadratic + ELVIRA",
    }
    color_dict = {
        "poly02half": cpink,
        "poly0_elvira": corange,
        "poly2_elvira": cred,
        "poly2h_elvira": cgreen,
    }

    generic_plot(data_manager,
                 # format=".pdf",
                 x="amplitude", y="error", label="updown_method",
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--", palette=updown_value_color_dict),
                 log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 error=get_reconstruction_error,
                 method=lambda models: names_dict[models],
                 # ylim=(1e-3, 1e0),
                 axes_by=["method"],
                 plot_by=["reconstruction_factor", "N", "frequency", "perturbation"])

    # generic_plot(data_manager,
    #              name="InterfaceError",
    #              # format=".pdf",
    #              x="amplitude", y="interface_error", label="updown_method",
    #              plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--", palette=updown_value_color_dict),
    #              log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
    #              interface_error=get_reconstruction_error_in_interface,
    #              method=lambda models: names_dict[models],
    #              # ylim=(1e-3, 1e0),
    #              axes_by=["method"],
    #              plot_by=["reconstruction_factor", "N", "frequency", "perturbation"])

    plot_reconstruction(
        data_manager,
        name="Reconstruction",
        folder='reconstruction',
        axes_by=["updown_method"],
        plot_by=['models', "amplitude"],
        folder_by=['image', "num_cells_per_dim", "reconstruction_factor", "frequency", "perturbation"],
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
        num_cores=15,
    )

    print("CO2 consumption: ", data_manager.CO2kg)
