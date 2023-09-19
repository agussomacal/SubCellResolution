import time
from functools import partial

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot
from experiments.image_reconstruction import plot_reconstruction
from experiments.subcell_paper.global_params import CCExtraWeight, EVALUATIONS
from experiments.subcell_paper.models2compare import elvira_cc, aero_q
from experiments.subcell_paper.tools import get_reconstruction_error, calculate_averages_from_image, load_image, \
    get_reconstruction_error_in_interface, reconstruct
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.RegularCellCreator import PolynomialRegularCellCreator, weight_cells_by_distance
from lib.CellIterators import iterate_all
from lib.CellOrientators import BaseOrientator
from lib.SmoothnessCalculators import oracle
from lib.StencilCreators import StencilCreatorFixedShape
from lib.StencilCreators import StencilCreatorSameRegionAdaptive
from lib.SubCellReconstruction import SubCellReconstruction, ReconstructionErrorMeasure, CellCreatorPipeline


def trigonometric(image, amplitude, frequency):
    image = load_image(image)
    h, w = np.shape(image)
    y, x = np.meshgrid(*list(map(range, np.shape(image))))
    d = np.sqrt((x - h / 2) ** 2 + (y - w / 2) ** 2)
    # v=128+(64+32*sin((x-w/2+y-h/2)*5*6/w))*(v >0)-(v==0)*(64+32*cos(d*5*6/w))
    image += amplitude * (
            (image >= 0.5) * np.cos(2 * np.pi * x * frequency / w) +
            (image <= 0.5) * np.sin(2 * np.pi * d * frequency / w)
    )

    return {
        "enhanced_image": image
    }


def parabolas(image, amplitude, frequency):
    image = load_image(image)
    h, w = np.shape(image)
    y, x = np.meshgrid(*list(map(range, np.shape(image))))
    d = np.sqrt((x - h / 2) ** 2 + (y - w / 2) ** 2)
    # v=128+(64+32*sin((x-w/2+y-h/2)*5*6/w))*(v >0)-(v==0)*(64+32*cos(d*5*6/w))
    image += amplitude * (
            (image >= 0.5) * (1 - 0.5 * (2 * np.pi * x * frequency / w) ** 2) +
            (image <= 0.5) * (2 * np.pi * d * frequency / w) ** 2
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
    # stencil_creator=StencilCreatorFixedShape(stencil_shape=(5, 5)),
    cell_creator=PolynomialRegularCellCreator(
        degree=2, noisy=False,
        weight_function=partial(weight_cells_by_distance, central_cell_importance=CCExtraWeight, distance_weight=0.5),
        # weight_function=partial(weight_cells_by_smoothness, central_cell_importance=CCExtraWeight, epsilon=1e-5,
        #                         delta=0.05),
        dimensionality=2, full_rank=False)
)
regular_deg2_same_region = CellCreatorPipeline(
    cell_iterator=iterate_all,
    # cell_iterator=partial(iterate_by_reconstruction_error_and_smoothness, value=REGULAR_CELL,
    #                       condition=operator.eq),  # only regular cells
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
    stencil_creator=StencilCreatorSameRegionAdaptive(num_nodes_per_dim=3),
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
def poly0_elvira(smoothness_calculator, angle_threshold=20):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_constant_same_region,
            elvira_cc(angle_threshold=angle_threshold),
        ],
        obera_iterations=0
    )


@fit_model
def poly1_elvira(smoothness_calculator, angle_threshold=20):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_deg1_same_region,
            elvira_cc(angle_threshold=angle_threshold),
        ],
        obera_iterations=0
    )


@fit_model
def poly2_elvira(smoothness_calculator, angle_threshold=20):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_deg2_same_region,
            elvira_cc(angle_threshold=angle_threshold),
        ],
        obera_iterations=0
    )


@fit_model
def poly02_elvira(smoothness_calculator, angle_threshold=20):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_constant_same_region,
            elvira_cc(angle_threshold=angle_threshold),
            regular_deg2_same_region,
        ],
        obera_iterations=0
    )


@fit_model
def poly2h_elvira(smoothness_calculator, angle_threshold=20):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_deg2half_same_region,
            elvira_cc(angle_threshold=angle_threshold),
        ],
        obera_iterations=0
    )


@fit_model
def poly02h_elvira(smoothness_calculator, angle_threshold=20):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            regular_constant_same_region,
            elvira_cc(angle_threshold=angle_threshold),
            regular_deg2half_same_region,

        ],
        obera_iterations=0
    )


@fit_model
def poly1_elvira(smoothness_calculator, angle_threshold=20):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        [
            regular_deg1_same_region,
            elvira_cc(angle_threshold=angle_threshold),
            aero_q(angle_threshold=angle_threshold),
            regular_deg2_same_region],
        obera_iterations=0
    )


@fit_model
def poly2_qelvira(smoothness_calculator, angle_threshold=20):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        # regular cell with piecewise_constant
        [
            regular_deg2_same_region,
            elvira_cc(angle_threshold=angle_threshold),
            aero_q(angle_threshold=angle_threshold),
        ],
        obera_iterations=0
    )


@fit_model
def poly02_qelvira(smoothness_calculator, angle_threshold=20):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        # regular cell with piecewise_constant
        [
            regular_constant_same_region,
            elvira_cc(angle_threshold=angle_threshold),
            aero_q(angle_threshold=angle_threshold),
            regular_deg2_same_region
        ],
        obera_iterations=0
    )


@fit_model
def poly02h_qelvira(smoothness_calculator, angle_threshold=20):
    return SubCellReconstruction(
        name="All",
        smoothness_calculator=smoothness_calculator,
        reconstruction_error_measure=reconstruction_error_measure,
        refinement=1,
        cell_creators=
        # regular cell with piecewise_constant
        [
            regular_constant_same_region,
            elvira_cc(angle_threshold=angle_threshold),
            aero_q(angle_threshold=angle_threshold),
            regular_deg2half_same_region
        ],
        obera_iterations=0
    )


if __name__ == "__main__":
    name = 'PieceWiseRegularQELVIRA'
    # name = 'PieceWiseRegularELVIRA'

    if name == 'PieceWiseRegularQELVIRA':
        models = [
            poly0_elvira,
            poly2_elvira,
            poly2_qelvira
        ]
    else:
        models = [
            poly0_elvira,
            poly1_elvira,
            poly2_elvira,
            poly02_elvira,
            poly2h_elvira,
            poly02h_elvira,
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
        parabolas
    )

    lab.define_new_block_of_functions(
        "models",
        *models,
        recalculate=False
    )

    lab.execute(
        data_manager,
        num_cores=10,
        recalculate=False,
        save_on_iteration=15,
        forget=True,
        frequency=[0.5, 2],
        # amplitude=[1e-5, 1e-4, 1e-3, 1e-2, 5e-2, 1e-1, 2e-1, 5e-1],
        amplitude=[1e-3, 1e-2, 1e-1],
        # num_cells_per_dim=[20, 40],
        num_cells_per_dim=[20],
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
                 # ylim=(1e-3, 1e0),
                 # axes_by=["frequency"],
                 plot_by=["reconstruction_factor", "N", "frequency", "perturbation"])

    generic_plot(data_manager,
                 name="InterfaceError",
                 x="amplitude", y="interface_error", label="models",
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 interface_error=get_reconstruction_error_in_interface,
                 # ylim=(1e-3, 1e0),
                 axes_by=[],
                 plot_by=["reconstruction_factor", "N", "frequency", "perturbation"])

    plot_reconstruction(
        data_manager,
        name="Reconstruction",
        folder='reconstruction',
        axes_by=["amplitude", ],
        plot_by=['models', ],
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
        num_cores=1,
    )

    print("CO2 consumption: ", data_manager.CO2kg)
