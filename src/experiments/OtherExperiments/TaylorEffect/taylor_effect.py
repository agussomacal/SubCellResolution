import time

import numpy as np
import pandas as pd
from experiments.tools import load_image

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.visualization import perplex_plot, generic_plot
from experiments.models import calculate_averages_from_image
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import ArrayIndexerNd


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


def image_reconstruction(enhanced_image, model, reduced_image_size_factor, num_cells_per_dim):
    # reduced_shape = np.array(np.shape(enhanced_image)) // reduced_image_size_factor
    # resolution_factor = reduced_shape // num_cells_per_dim
    #
    # enhanced_image = calculate_averages_from_image(enhanced_image, reduced_shape)

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


@perplex_plot
def plot_reconstruction(fig, ax, enhanced_image, num_cells_per_dim, model, reconstruction, alpha=0.5,
                        plot_original_image=True,
                        difference=False, plot_curve=True, plot_curve_winner=False, plot_vh_classification=True,
                        plot_singular_cells=True, cmap="magma", trim=((0, 0), (0, 0)), numbers_on=True, *args,
                        **kwargs):
    image = enhanced_image.pop()
    num_cells_per_dim = num_cells_per_dim.pop()
    model = model.pop()
    reconstruction = reconstruction.pop()

    model_resolution = np.array(model.resolution)
    # image = load_image(image)

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
        num_cores=15,
        recalculate=False,
        forget=False,
        save_on_iteration=5,
        refinement=[1],
        # num_cells_per_dim=[42*2],  # , 28, 42
        # num_cells_per_dim=[28, 42, 42 * 2],  # , 28, 42
        num_cells_per_dim=[
            28,
            42 * 2
        ],  # , 28, 42
        # num_cells_per_dim=[42],  # , 28, 42
        noise=[0],
        amplitude=[0, 1e-1, 5e-1],
        reduced_image_size_factor=[6],
        # amplitude=[0, 1e-3, 1e-2, 1e-1, 0.5, 1],
        image=[
            "Elipsoid_1680x1680.jpg"
        ]
    )

    generic_plot(data_manager, x="N", y="error", label="models",
                 N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 error=lambda reconstruction_error: np.sqrt(np.mean(reconstruction_error ** 2)),
                 axes_by=["amplitude"],
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
        num_cores=1,
        reduced_image_size_factor=6
    )
