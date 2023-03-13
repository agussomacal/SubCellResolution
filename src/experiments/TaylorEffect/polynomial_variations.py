import time
from functools import partial

import matplotlib.pylab as plt
import numpy as np
import pandas as pd

import config
from experiments.TaylorEffect.taylor_effect import enhance_image, image_reconstruction
from experiments.image_reconstruction import plot_reconstruction
from experiments.models import load_image, calculate_averages_from_image
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_cells_by_grad
from lib.CellCreators.RegularCellCreator import PolynomialRegularCellCreator, weight_cells, weight_cells_extra_weight
from lib.CellIterators import iterate_all, iterate_by_smoothness
from lib.CellOrientators import BaseOrientator, OrientByGradient
from lib.SmoothnessCalculators import indifferent, by_gradient
from lib.StencilCreators import StencilCreatorSameRegionAdaptive, StencilCreatorFixedShape, \
    StencilCreatorSmoothnessDistTradeOff
from lib.SubCellReconstruction import CellCreatorPipeline, SubCellReconstruction, ReconstructionErrorMeasureBase, \
    ReconstructionErrorMeasure
from src.DataManager import DataManager, JOBLIB
from src.Indexers import ArrayIndexerNd
from src.LabPipeline import LabPipeline
from src.viz_utils import perplex_plot


def fit_model_decorator(function):
    def decorated_func(enhanced_image, num_cells_per_dim, noise, refinement, dist_trade_off, avg_diff_trade_off):
        np.random.seed(42)
        avg_values = calculate_averages_from_image(enhanced_image, num_cells_per_dim)
        avg_values += np.random.uniform(-noise, noise, size=avg_values.shape)

        model = function(refinement, dist_trade_off, avg_diff_trade_off)

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


@fit_model_decorator
def polynomial2_adapt(refinement: int, dist_trade_off, avg_diff_trade_off):
    return SubCellReconstruction(
        name="Polynomial2",
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
                stencil_creator=StencilCreatorSmoothnessDistTradeOff(stencil_shape=(3, 3),
                                                                     dist_trade_off=dist_trade_off,
                                                                     avg_diff_trade_off=avg_diff_trade_off),
                cell_creator=PolynomialRegularCellCreator(
                    degree=2, dimensionality=2, noisy=False, full_rank=False,
                    weight_function=partial(weight_cells_extra_weight, central_cell_extra_weight=100)
                )
            )
        ]
    )


@perplex_plot
def plot_convergence_curves(fig, ax, amplitude, reconstruction_error, dist_trade_off, avg_diff_trade_off):
    data = pd.DataFrame.from_dict({
        "distance_weight": dist_trade_off,
        "average_weight": avg_diff_trade_off,
        "Error": list(map(np.mean, reconstruction_error)),
    })
    data.sort_values(by=["average_weight", "distance_weight"], inplace=True)
    # data.sort_values(by="Model", inplace=True)

    for avg_w, d in data.groupby("avg_diff_trade_off"):
        plt.plot(d["distance_weight"], d["Error"], ".-", label=avg_w)

    ax.set_ylabel("L1 reconstruction error")
    ax.set_xlabel("Distance weight")

    # ax.set_xticks(data["perturbation"], data["perturbation"])
    # y_ticks = np.arange(1 - int(np.log10(data["Error"].min())))
    # ax.set_yticks(10.0 ** (-y_ticks), [fr"$10^{-y}$" for y in y_ticks])
    # ax.set_yscale("log")
    # ax.set_xscale("log")
    ax.legend()


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='PolynomialVariations',
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
        polynomial2_adapt,
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
        dist_trade_off=[0, 0.25, 0.5, 1],
        avg_diff_trade_off=[0, 0.25, 0.5, 1],
        # num_cells_per_dim=[42*2],  # , 28, 42
        # num_cells_per_dim=[28, 42, 42 * 2],  # , 28, 42
        num_cells_per_dim=[
            28,
            # 42 * 2
        ],  # , 28, 42
        # num_cells_per_dim=[42],  # , 28, 42
        noise=[0],
        # amplitude=[0, 1],
        amplitude=[1],
        image=[
            "Elipsoid_1680x1680.jpg"
        ]
    )

    plot_convergence_curves(data_manager,
                            axes_by=[],
                            plot_by=["num_cells_per_dim", "image", "refinement"])
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
