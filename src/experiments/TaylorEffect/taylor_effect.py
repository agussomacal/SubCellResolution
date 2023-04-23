import time
from functools import partial

import numpy as np
import pandas as pd

import config
from experiments.image_reconstruction import plot_reconstruction
from experiments.models import calculate_averages_from_image
from experiments.subcell_paper.function_families import load_image
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_cells_by_grad
from lib.CellCreators.RegularCellCreator import PolynomialRegularCellCreator, weight_cells_extra_weight
from lib.CellIterators import iterate_all, iterate_by_smoothness
from lib.CellOrientators import BaseOrientator, OrientByGradient
from lib.SmoothnessCalculators import indifferent, by_gradient
from lib.StencilCreators import StencilCreatorSameRegionAdaptive, StencilCreatorFixedShape, \
    StencilCreatorSmoothnessDistTradeOff
from lib.SubCellReconstruction import CellCreatorPipeline, SubCellReconstruction, ReconstructionErrorMeasureBase, \
    ReconstructionErrorMeasure
from src.DataManager import DataManager, JOBLIB
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from src.LabPipeline import LabPipeline
from src.visualization import perplex_plot, generic_plot


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


def fit_model_decorator(function):
    def decorated_func(enhanced_image, num_cells_per_dim, noise, refinement):
        np.random.seed(42)
        avg_values = calculate_averages_from_image(enhanced_image, num_cells_per_dim)
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


def image_reconstruction(enhanced_image, model):
    t0 = time.time()
    reconstruction = model.reconstruct_arbitrary_size(np.shape(enhanced_image))
    # reconstruction = model.reconstruct_by_factor(
    #     resolution_factor=np.array(np.array(np.shape(enhanced_image)) / np.array(model.resolution), dtype=int))
    t_reconstruct = time.time() - t0

    reconstruction_error = np.abs(np.array(reconstruction) - enhanced_image)
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
                # stencil_creator=StencilCreatorSmoothnessDistTradeOff(stencil_shape=(3, 3), dist_trade_off=0.5,
                #                                                      avg_diff_trade_off=1),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 3)),
                cell_creator=PolynomialRegularCellCreator(
                    degree=2, dimensionality=2, noisy=False, full_rank=False,
                    weight_function=None
                    # partial(weight_cells, central_cell_extra_weight=100)
                )
            )
        ]
    )


@fit_model_decorator
def polynomial2_adapt(refinement: int):
    return SubCellReconstruction(
        name="Polynomial2_adapt",
        smoothness_calculator=by_gradient,
        reconstruction_error_measure=ReconstructionErrorMeasure(
            stencil_creator=StencilCreatorFixedShape((3, 3)), central_cell_extra_weight=1, metric="l2"),
        refinement=refinement,
        cell_creators=
        [
            # regular cell with quadratics
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # all cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorSmoothnessDistTradeOff(stencil_shape=(3, 3), dist_trade_off=0.5,
                                                                     avg_diff_trade_off=0.1),
                cell_creator=PolynomialRegularCellCreator(
                    degree=2, dimensionality=2, noisy=False, full_rank=False,
                    weight_function=partial(weight_cells_extra_weight, central_cell_extra_weight=100)
                )
            )
        ]
    )


@fit_model_decorator
def polynomial2_fix(refinement: int):
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
                # stencil_creator=StencilCreatorSmoothnessDistTradeOff(stencil_shape=(3, 3), dist_trade_off=0.5,
                #                                                      avg_diff_trade_off=1),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 3)),
                cell_creator=PolynomialRegularCellCreator(
                    degree=2, dimensionality=2, noisy=False, full_rank=False,
                    weight_function=partial(weight_cells_extra_weight, central_cell_extra_weight=100)
                )
            )
        ]
    )


@fit_model_decorator
def elvira(refinement: int):
    return SubCellReconstruction(
        name="ELVIRA",
        smoothness_calculator=by_gradient,
        reconstruction_error_measure=ReconstructionErrorMeasure(
            stencil_creator=StencilCreatorFixedShape((3, 3)), central_cell_extra_weight=1, metric="l2"),
        refinement=refinement,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # all cells
                orientator=BaseOrientator(dimensionality=2),
                # stencil_creator=StencilCreatorSmoothnessDistTradeOff(stencil_shape=(3, 3), dist_trade_off=0.5,
                #                                                      avg_diff_trade_off=1),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 3)),
                cell_creator=PolynomialRegularCellCreator(
                    degree=2, dimensionality=2, noisy=False, full_rank=False,
                    weight_function=partial(weight_cells_extra_weight, central_cell_extra_weight=100)
                )
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


@perplex_plot
def plot_convergence_curves(fig, ax, amplitude, reconstruction_error, models):
    data = pd.DataFrame.from_dict({
        "perturbation": np.array(amplitude) + 1,
        # "N": np.array(num_cells_per_dim)**2,
        "Error": list(map(np.mean, reconstruction_error)),
        "Model": models
    })
    data.sort_values(by=["Model", "perturbation"], inplace=True)
    # data.sort_values(by="Model", inplace=True)

    for model, d in data.groupby("Model"):
        ax.plot(d["perturbation"], d["Error"], ".-", label=model)

    ax.set_ylabel("L1 reconstruction error")
    ax.set_xlabel("Regular perturbation amplitude")

    ax.set_xticks(data["perturbation"], data["perturbation"])
    y_ticks = np.arange(1 - int(np.log10(data["Error"].min())))
    ax.set_yticks(10.0 ** (-y_ticks), [fr"$10^{-y}$" for y in y_ticks])
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='TaylorEffect2',
        format=JOBLIB
    )
    data_manager.load()

    lab = LabPipeline()

    lab.define_new_block_of_functions(
        "perturbation",
        enhance_image
    )

    lab.define_new_block_of_functions(
        "models",
        # polynomial2,
        piecewise_constant,
        polynomial2_fix,
        polynomial2_adapt,
        elvira
    )

    lab.define_new_block_of_functions(
        "image_reconstruction",
        image_reconstruction
    )

    lab.execute(
        data_manager,
        num_cores=3,
        recalculate=False,
        forget=False,
        refinement=[1],
        # num_cells_per_dim=[42*2],  # , 28, 42
        # num_cells_per_dim=[28, 42, 42 * 2],  # , 28, 42
        num_cells_per_dim=[
            28,
            42 * 2
        ],  # , 28, 42
        # num_cells_per_dim=[42],  # , 28, 42
        noise=[0],
        # amplitude=[0, 1],
        amplitude=[0, 1e-3, 1e-2, 1e-1, 0.5, 1],
        image=[
            "Elipsoid_1680x1680.jpg"
        ]
    )

    generic_plot(data_manager, x="N", y="error", label="amplitude",
                 N=lambda num_cells_per_dim: num_cells_per_dim**2,
                 error=lambda reconstruction_error: np.sqrt(np.mean(reconstruction_error**2)),
                 axes_by=["models"],
                 plot_by=["image", "refinement"])
    # plot_convergence_curves(data_manager,
    #                         axes_by=[],
    #                         plot_by=["num_cells_per_dim", "image", "refinement"])

    plot_reconstruction(
        data_manager,
        name="BackgroundImage",
        folder='reconstruction',
        axes_by=["amplitude"],
        plot_by=['models', 'image', 'num_cells_per_dim', 'refinement'],
        axes_xy_proportions=(15, 15),
        difference=False,
        plot_curve=True,
        plot_curve_winner=False,
        plot_vh_classification=False,
        plot_singular_cells=False,
        plot_original_image=True,
        numbers_on=True,
        plot_again=True,
        num_cores=1
    )
