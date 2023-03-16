import time
from functools import partial

import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
import pandas as pd

import config
from experiments.TaylorEffect.taylor_effect import enhance_image, image_reconstruction
from experiments.VizReconstructionUtils import plot_cells, draw_cell_borders
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
from src.viz_utils import perplex_plot, generic_plot


def core_experiment(enhanced_image, num_cells_per_dim, noise, dist_trade_off, central_cell_importance, delta, epsilon,
                    degree=2):
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
                    degree=degree, dimensionality=2, noisy=False, full_rank=False,
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
    t_reconstruct = time.time() - t0
    return t_fit, t_reconstruct, reconstruction, model


def experiment(enhanced_image, num_cells_per_dim, noise, dist_trade_off, central_cell_importance, delta, epsilon,
               degree=2):
    t_fit, t_reconstruct, reconstruction, _ = core_experiment(enhanced_image, num_cells_per_dim, noise, dist_trade_off,
                                                              central_cell_importance, delta, epsilon,
                                                              degree)

    return {
        # "model": model,
        "time_to_fit": t_fit,
        "time_to_reconstruct": t_reconstruct,
        "mse": np.sqrt(np.mean((reconstruction - enhanced_image) ** 2))
        # "reconstruction": reconstruction
    }


@perplex_plot
def plot_reconstruction(fig, ax, enhanced_image, num_cells_per_dim, noise, dist_trade_off, central_cell_importance,
                        delta, epsilon, degree, alpha=0.5, plot_original_image=True, difference=False, cmap="magma",
                        trim=((0, 0), (0, 0)), numbers_on=True, *args, **kwargs):
    image = enhanced_image.pop()
    num_cells_per_dim = num_cells_per_dim.pop()
    noise = noise.pop()
    dist_trade_off = dist_trade_off.pop()
    central_cell_importance = central_cell_importance.pop(),
    delta = delta.pop()
    epsilon = epsilon.pop()
    degree = degree.pop()
    _, _, reconstruction, model = core_experiment(
        image, num_cells_per_dim, noise, dist_trade_off, central_cell_importance, delta, epsilon, degree)

    model_resolution = np.array(model.resolution)

    if plot_original_image:
        plot_cells(ax, colors=image, mesh_shape=model_resolution, alpha=alpha, cmap="Greys_r",
                   vmin=np.min(image), vmax=np.max(image))

    if difference:
        plot_cells(ax, colors=reconstruction - image, mesh_shape=model_resolution, alpha=alpha, cmap=cmap, vmin=-1,
                   vmax=1)
    else:
        plot_cells(ax, colors=reconstruction, mesh_shape=model_resolution, alpha=alpha, cmap=cmap, vmin=-1, vmax=1)

    draw_cell_borders(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )
    ax.set_xlim((-0.5 + trim[0][0], model.resolution[0] - trim[0][1] - 0.5))
    ax.set_ylim((model.resolution[1] - trim[1][0] - 0.5, trim[1][1] - 0.5))


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
        num_cores=15,
        recalculate=False,
        forget=False,
        save_on_iteration=None,
        degree=[2],
        # dist_trade_off=[0, 0.5, 0.8, 1],
        dist_trade_off=[0, 0.8, 1],
        central_cell_importance=[0, 100],
        # delta=[0, 0.05, 0.5],
        delta=[0.05],
        # epsilon=[1e-5, 1e5],
        epsilon=[1e-5, 1e5],
        num_cells_per_dim=[
            28  # , 42 * 2
        ],
        noise=[0],
        amplitude=[0, 1e-2, 1e-1, 1],
        image=[
            "Elipsoid_1680x1680.jpg"
        ]
    )

    # plot_reconstruction(data_manager, amplitude=1e-1, num_cells_per_dim=28, noise=0, dist_trade_off=1,
    #                     central_cell_importance=0, delta=0, epsilon=1e-5, degree=2,
    #                     alpha=0.5, plot_original_image=True, difference=False, cmap="magma",
    #                     trim=((0, 0), (0, 0)), numbers_on=True)

    generic_plot(data_manager, x="dist_trade_off", y="mse", label="label_var", plot_fun=sns.lineplot,
                 other_plot_funcs=(), log="y",
                 # mse=lambda reconstruction, enhanced_image: np.sqrt(np.mean((reconstruction-enhanced_image)**2)),
                 label_var=lambda central_cell_importance,
                                  delta: f"central_cell_importance: {central_cell_importance}, delta: {delta}",
                 axes_by=["epsilon", "num_cells_per_dim"],
                 plot_by=["amplitude"])

    generic_plot(data_manager, x="amplitude", y="mse", label="label_var", plot_fun=sns.lineplot,
                 other_plot_funcs=(), log="y",
                 # mse=lambda reconstruction, enhanced_image: np.sqrt(np.mean((reconstruction-enhanced_image)**2)),
                 label_var=lambda dist_trade_off, delta: f"dist weight: {dist_trade_off}, delta: {delta}",
                 axes_by=["central_cell_importance", "epsilon"])
