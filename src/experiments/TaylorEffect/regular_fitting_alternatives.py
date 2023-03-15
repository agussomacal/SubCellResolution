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
from lib.CellCreators.RegularCellCreator import PolynomialRegularCellCreator, weight_cells, weight_cells_extra_weight, \
    weight_cells_by_smoothness
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


def experiment(enhanced_image, num_cells_per_dim, noise, dist_trade_off, central_cell_importance, delta, epsilon):
    model = SubCellReconstruction(
        name="Polynomial",
        smoothness_calculator=by_gradient,
        reconstruction_error_measure=ReconstructionErrorMeasure(
            stencil_creator=StencilCreatorFixedShape((3, 3)), central_cell_extra_weight=1, metric="l2"),
        refinement=1,
        cell_creators=
        [
            # regular cell with quadratics
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # all cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorSmoothnessDistTradeOff(stencil_shape=(3, 3),
                                                                     dist_trade_off=dist_trade_off),
                cell_creator=PolynomialRegularCellCreator(
                    degree=2, dimensionality=2, noisy=False, full_rank=False,
                    weight_function=partial(weight_cells_by_smoothness,
                                            central_cell_importance=central_cell_importance,
                                            delta=delta, epsilon=epsilon)
                )
            )
        ]
    )

    np.random.seed(42)
    avg_values = calculate_averages_from_image(enhanced_image, num_cells_per_dim)
    avg_values += np.random.uniform(-noise, noise, size=avg_values.shape)

    t0 = time.time()
    model.fit(average_values=avg_values,
              indexer=ArrayIndexerNd(avg_values, "cyclic"))
    t_fit = time.time() - t0

    t0 = time.time()
    reconstruction = model.reconstruct_arbitrary_size(np.shape(enhanced_image))
    # reconstruction = model.reconstruct_by_factor(
    #     resolution_factor=np.array(np.array(np.shape(enhanced_image)) / np.array(model.resolution), dtype=int))
    t_reconstruct = time.time() - t0

    return {
        "model": model,
        "time_to_fit": t_fit,
        "time_to_reconstruct": t_reconstruct,
        "reconstruction": reconstruction
    }



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
        name='RegularFitAlternatives',
        format=JOBLIB
    )
    data_manager.load()

    lab = LabPipeline()

    lab.define_new_block_of_functions(
        "perturbation",
        enhance_image
    )

    lab.define_new_block_of_functions(
        "image_reconstruction",
        experiment
    )

    lab.execute(
        data_manager,
        num_cores=2,
        recalculate=False,
        forget=False,
        dist_trade_off=[0, 0.5, 0.8, 1],
        central_cell_importance=[0, 100],
        delta=[0, 0.05, 0.5],
        epsilon=[1e-5, 1e5],
        num_cells_per_dim=[
            28,
        ],
        noise=[0, 1e-2, 1e-1],
        amplitude=[1e-2, 1e-1, 1],
        image=[
            "Elipsoid_1680x1680.jpg"
        ],
        save_on_iteration=5
    )

    # generic_plot(data_manager, x, y, label, plot_funcsns.lineplot,
    #              other_plot_funcs = (), log= "", ** kwargs)
    # plot_convergence_curves(data_manager,
    #                         axes_by=[],
    #                         plot_by=["num_cells_per_dim", "image", "refinement"])
    # plot_reconstruction(
    #     data_manager,
    #     name="BackgroundImage",
    #     folder='reconstruction',
    #     axes_by=["amplitude"],
    #     plot_by=['models', 'image', 'num_cells_per_dim', 'refinement'],
    #     axes_xy_proportions=(15, 15),
    #     difference=False,
    #     plot_curve=True,
    #     plot_curve_winner=False,
    #     plot_vh_classification=False,
    #     plot_singular_cells=False,
    #     plot_original_image=True,
    #     numbers_on=True,
    #     plot_again=True,
    #     num_cores=1
    # )
