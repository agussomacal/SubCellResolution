import time
from functools import partial

import numpy as np

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.visualization import generic_plot
from PerplexityLab.visualization import perplex_plot, one_line_iterator
from experiments.VizReconstructionUtils import plot_cells, draw_cell_borders, \
    plot_cells_not_regular_classification_core, plot_cells_vh_classification_core, plot_curve_core, plot_cells_identity
from experiments.models import calculate_averages_from_image
from experiments.tools import load_image
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE
from lib.CellCreators.RegularCellCreator import MirrorCellCreator, \
    PolynomialRegularCellCreator, weight_cells_by_smoothness
from lib.CellCreators.WENOCellCreators import WENO16RegularCellCreator, WENO1DRegularCellCreator, \
    WENO1DPointsRegularCellCreator
from lib.CellIterators import iterate_all
from lib.CellOrientators import BaseOrientator
from lib.SmoothnessCalculators import indifferent
from lib.StencilCreators import StencilCreatorFixedShape
from lib.SubCellReconstruction import SubCellReconstruction, CellCreatorPipeline, ReconstructionErrorMeasureBase


@perplex_plot()
@one_line_iterator()
def plot_reconstruction(fig, ax, image, num_cells_per_dim, model, reconstruction, alpha=0.5, plot_original_image=True,
                        difference=False, plot_curve=True, plot_curve_winner=False, plot_vh_classification=True,
                        plot_singular_cells=True, cmap="magma", trim=((0, 0), (0, 0)), numbers_on=True):
    model_resolution = np.array(model.resolution)
    image = load_image(image)

    if plot_original_image:
        plot_cells(ax, colors=image, mesh_shape=model_resolution, alpha=alpha, cmap="Greys_r",
                   vmin=np.min(image), vmax=np.max(image))

    if difference:
        plot_cells(ax, colors=reconstruction - image, mesh_shape=model.resolution, alpha=alpha, cmap=cmap, vmin=-1,
                   vmax=1)
    else:
        plot_cells(ax, colors=reconstruction, mesh_shape=model.resolution, alpha=alpha, cmap=cmap, vmin=-1, vmax=1)

    if plot_curve:
        if plot_curve_winner:
            plot_cells_identity(ax, model.resolution, model.cells, alpha=0.8)
            # plot_cells_type_of_curve_core(ax, model.resolution, model.cells, alpha=0.8)
        elif plot_vh_classification:
            plot_cells_vh_classification_core(ax, model.resolution, model.cells, alpha=0.8)
        elif plot_singular_cells:
            plot_cells_not_regular_classification_core(ax, model.resolution, model.cells, alpha=0.8)
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


def get_sub_cell_model(cell_creator, stencil_creator, refinement, name):
    return SubCellReconstruction(
        name=name,
        smoothness_calculator=indifferent,
        reconstruction_error_measure=ReconstructionErrorMeasureBase,
        refinement=refinement,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=stencil_creator,
                cell_creator=cell_creator
            ),
        ],
        obera_iterations=0
    )


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
    def decorated_func(enhanced_image, num_cells_per_dim, noise, refinement):
        np.random.seed(42)
        avg_values = calculate_averages_from_image(enhanced_image, num_cells_per_dim)
        avg_values += np.random.uniform(-noise, noise, size=avg_values.shape)

        model = sub_cell_model(refinement)

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
def piecewise_constant(refinement: int):
    return get_sub_cell_model(MirrorCellCreator(dimensionality=2), StencilCreatorFixedShape(stencil_shape=(1, 1)),
                              refinement, "PiecewiseConstant")


@fit_model
def fixed_polynomial_degree1(refinement: int):
    return get_sub_cell_model(PolynomialRegularCellCreator(degree=2, noisy=False, weight_function=None,
                                                           dimensionality=2, full_rank=True),
                              StencilCreatorFixedShape(stencil_shape=(3, 3)),
                              refinement, "FixedPolynomialDegree1")


@fit_model
def fixed_polynomial_degree2(refinement: int):
    return get_sub_cell_model(PolynomialRegularCellCreator(degree=2, noisy=False, weight_function=None,
                                                           dimensionality=2, full_rank=True),
                              StencilCreatorFixedShape(stencil_shape=(3, 3)),
                              refinement, "FixedPolynomialDegree2")


@fit_model
def fixed_polynomial_degree2_strict(refinement: int):
    return get_sub_cell_model(PolynomialRegularCellCreator(degree=2, noisy=False, weight_function=None,
                                                           dimensionality=2, full_rank=False),
                              StencilCreatorFixedShape(stencil_shape=(3, 3)),
                              refinement, "FixedPolynomialDegree2Strict")


@fit_model
def fixed_polynomial_degree2_weighted(refinement: int):
    return get_sub_cell_model(PolynomialRegularCellCreator(
        degree=2, noisy=False,
        weight_function=partial(weight_cells_by_smoothness, central_cell_importance=100, epsilon=1e-5, delta=0.05),
        dimensionality=2, full_rank=True),
        StencilCreatorFixedShape(stencil_shape=(3, 3)),
        refinement, "FixedPolynomialDegree2Weighted")


@fit_model
def fixed_polynomial_degree2_strict_weighted(refinement: int):
    return get_sub_cell_model(PolynomialRegularCellCreator(
        degree=2, noisy=False,
        weight_function=partial(weight_cells_by_smoothness, central_cell_importance=100, epsilon=1e-5, delta=0.05),
        dimensionality=2, full_rank=False),
        StencilCreatorFixedShape(stencil_shape=(3, 3)),
        refinement, "FixedPolynomialDegree2StrictWeighted")


@fit_model
def weno16(refinement: int):
    return get_sub_cell_model(WENO16RegularCellCreator(degree=1),
                              StencilCreatorFixedShape(stencil_shape=(5, 5)),
                              refinement, "WENO16")


@fit_model
def weno16_deg2(refinement: int):
    return get_sub_cell_model(WENO16RegularCellCreator(degree=2),
                              StencilCreatorFixedShape(stencil_shape=(5, 5)),
                              refinement, "WENO16_deg2")


@fit_model
def weno1d_4(refinement: int):
    return get_sub_cell_model(WENO1DRegularCellCreator(num_coeffs=4),
                              StencilCreatorFixedShape(stencil_shape=(5, 5)),
                              refinement, "WENO1D4")


@fit_model
def weno1d_5(refinement: int):
    return get_sub_cell_model(WENO1DRegularCellCreator(num_coeffs=5),
                              StencilCreatorFixedShape(stencil_shape=(5, 5)),
                              refinement, "WENO1D5")


@fit_model
def weno1d_9(refinement: int):
    return get_sub_cell_model(WENO1DRegularCellCreator(num_coeffs=9),
                              StencilCreatorFixedShape(stencil_shape=(5, 5)),
                              refinement, "WENO1D9")


@fit_model
def weno1d_points_9(refinement: int):
    return get_sub_cell_model(WENO1DPointsRegularCellCreator(num_coeffs=9),
                              StencilCreatorFixedShape(stencil_shape=(5, 5)),
                              refinement, "WENO1DPoints_9")


def image_reconstruction(enhanced_image, model, reconstruction_factor):
    t0 = time.time()
    reconstruction = model.reconstruct_arbitrary_size(np.array(np.shape(enhanced_image)) // reconstruction_factor)
    # reconstruction = model.reconstruct_by_factor(
    #     resolution_factor=np.array(np.array(np.shape(image)) / np.array(model.resolution), dtype=int))
    t_reconstruct = time.time() - t0

    if reconstruction_factor > 1:
        # TODO: should be the evaluations not the averages.
        enhanced_image = calculate_averages_from_image(enhanced_image, num_cells_per_dim=np.shape(reconstruction))
    reconstruction_error = np.abs(np.array(reconstruction) - enhanced_image)
    return {
        "reconstruction": reconstruction,
        "reconstruction_error": reconstruction_error,
        "time_to_reconstruct": t_reconstruct
    }


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='WENO2',
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
        # fixed_polynomial_degree2,
        # fixed_polynomial_degree2_strict,
        # fixed_polynomial_degree2_strict_weighted,
        # fixed_polynomial_degree2_weighted,

        # piecewise_constant,
        # fixed_polynomial_degree1,
        # weno16,
        # weno16_deg2,

        weno1d_4,
        weno1d_5,
        # weno1d_9
        weno1d_points_9
    )

    lab.define_new_block_of_functions(
        "image_reconstruction",
        image_reconstruction
    )

    lab.execute(
        data_manager,
        num_cores=15,
        recalculate=False,
        forget=False,
        # amplitude=[0, 1e-3, 1e-2, 2e-2, 5e-2, 1e-1, 2e-1, 5e-1, 1],
        amplitude=[1e-3, 1e-2, 5e-2, 1e-1, 1],
        refinement=[1],
        # num_cells_per_dim=[14, 20, 28, 42, 42 * 2],  # 42 * 2
        num_cells_per_dim=[20, 42],  # 42 * 2
        noise=[0],
        image=[
            "Ellipsoid_1680x1680.png",
        ],
        reconstruction_factor=[6],
        # reconstruction_factor=[6],
    )

    generic_plot(data_manager, x="amplitude", y="mse", label="models",
                 # plot_func=NamedPartial(sns.lineplot, marker=".", linestyle="--"),
                 log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=lambda reconstruction_error: np.mean(reconstruction_error),
                 axes_by=["num_cells_per_dim"],
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
        plot_by=['image', "num_cells_per_dim", 'refinement', "amplitude", "reconstruction_factor"],
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
        amplitude=1
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
