import matplotlib.pylab as plt
import numpy as np
import pandas as pd

import config
from experiments.VizReconstructionUtils import plot_cells, draw_cell_borders, \
    plot_cells_not_regular_classification_core, plot_cells_vh_classification_core, plot_curve_core, plot_cells_identity
from experiments.models import calculate_averages_from_image, load_image, image_reconstruction, polynomial2, elvira_go
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.visualization import perplex_plot


@perplex_plot
def plot_convergence_curves(fig, ax, num_cells_per_dim, reconstruction_error, models):
    data = pd.DataFrame.from_dict({
        "N": np.array(num_cells_per_dim) ** 2,
        "Error": list(map(np.mean, reconstruction_error)),
        "Model": models
    })
    for model, d in data.groupby("Model"):
        plt.plot(d["N"], d["Error"], label=model)
    ax.set_xticks(num_cells_per_dim, num_cells_per_dim)
    y_ticks = np.arange(1 - int(np.log10(data["Error"].min())))
    ax.set_yticks(10.0 ** (-y_ticks), [fr"$10^{-y}$" for y in y_ticks])
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()


@perplex_plot
def plot_reconstruction(fig, ax, image, num_cells_per_dim, model, reconstruction, alpha=0.5, plot_original_image=True,
                        difference=False, plot_curve=True, plot_curve_winner=False, plot_vh_classification=True,
                        plot_singular_cells=True, cmap="magma", trim=((0, 0), (0, 0)), numbers_on=True, *args,
                        **kwargs):
    image = image.pop()
    num_cells_per_dim = num_cells_per_dim.pop()
    model = model.pop()
    reconstruction = reconstruction.pop()

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


@perplex_plot
def plot_fast_reconstruction(fig, ax, image, num_cells_per_dim, model, alpha=0.5, resolution_factor: int = 3,
                             difference=False, plot_curve=True, plot_curve_winner=False, plot_vh_classification=True,
                             plot_singular_cells=True, plot_original_image=True, cmap="magma", trim=((0, 0), (0, 0)),
                             numbers_on=True, *args, **kwargs):
    image = image.pop()
    num_cells_per_dim = num_cells_per_dim.pop()
    model = model.pop()

    model_resolution = np.array(model.resolution)
    image = load_image(image)
    im_shape = np.array(np.shape(image))

    resolution_factor = np.array(
        (im_shape / model_resolution) // ((im_shape / model_resolution) / resolution_factor),
        dtype=int)

    if plot_original_image:
        plot_cells(ax, colors=image, mesh_shape=model_resolution, alpha=alpha, cmap="Greys_r",
                   vmin=np.min(image), vmax=np.max(image))
    image = calculate_averages_from_image(image, model_resolution * resolution_factor)
    reconstruction = model.reconstruct_by_factor(resolution_factor=resolution_factor)

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


@perplex_plot
def plot_original_image(fig, ax, image, num_cells_per_dim, model, alpha=0.5, cmap="Greys_r", trim=((0, 0), (0, 0)),
                        numbers_on=True, averages=False, *args, **kwargs):
    image = image.pop()
    num_cells_per_dim = num_cells_per_dim.pop()
    model = model.pop()

    model_resolution = np.array(model.resolution)
    image = load_image(image)
    if averages:
        image = calculate_averages_from_image(image, num_cells_per_dim)

    plot_cells(ax, colors=image, mesh_shape=model_resolution, alpha=alpha, cmap=cmap,
               vmin=np.min(image), vmax=np.max(image))

    draw_cell_borders(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )
    ax.set_xlim((-0.5 + trim[0][0], model_resolution[0] - trim[0][1] - 0.5))
    ax.set_ylim((model_resolution[1] - trim[1][0] - 0.5, trim[1][1] - 0.5))


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='ImageReconstruction',
        format=JOBLIB
    )
    # data_manager.load()

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "models",
        polynomial2,
        # piecewise_constant,
        # elvira,
        elvira_go
    )

    lab.define_new_block_of_functions(
        "image_reconstruction",
        image_reconstruction
    )

    lab.execute(
        data_manager,
        num_cores=15,
        recalculate=False,
        forget=True,
        refinement=[1],
        num_cells_per_dim=[42 * 2],  # , 28, 42
        # num_cells_per_dim=[28],  # , 28, 42
        # num_cells_per_dim=[42],  # , 28, 42
        noise=[0],
        image=[
            "ShapesVertex_1680x1680.jpg",
            # "ShapesVertexRegular_1680x1680.png",
            # v=128+(64+32*sin((x-w/2+y-h/2)*5*6/w))*(v >0)-(v==0)*(64+32*cos(d*5*6/w))
            "peppers.jpg",
            "R2D2.jpeg",
            "DarthVader.jpeg",
            # "mountains_WB.jpg",
            # "House_in_the_sea.jpg",
            # "RegularShock.png"  # v=128*cos(d/100)+128*x/w+128*((x*x+y*y)<h*h/4)-128*(x>(w*2/3))
        ]
    )

    # plot_convergence_curves(data_manager)
    # plot_fast_reconstruction(
    #     data_manager,
    #     name="BackgroundImage",
    #     folder='reconstruction',
    #     image="ShapesVertexRegular_1680x1680.png",
    #     axes_by=[],
    #     plot_by=['image', 'models', 'num_cells_per_dim', 'refinement'],
    #     axes_xy_proportions=(15, 15),
    #     resolution_factor=5,
    #     difference=False,
    #     plot_curve=True,
    #     plot_curve_winner=False,
    #     plot_vh_classification=False,
    #     plot_singular_cells=False,
    #     plot_original_image=True,
    #     numbers_on=True,
    #     plot_again=True,
    #     num_cores=1,
    #     reduced_image_size_factor=6
    # )

    plot_fast_reconstruction(
        data_manager,
        name="Reconstruction",
        folder='reconstruction',
        # image="ShapesVertexRegular_1680x1680.png",
        axes_by=[],
        plot_by=['image', 'models', 'num_cells_per_dim', 'refinement'],
        axes_xy_proportions=(15, 15),
        resolution_factor=5,
        difference=False,
        plot_curve=True,
        plot_curve_winner=False,
        plot_vh_classification=False,
        plot_singular_cells=False,
        plot_original_image=False,
        numbers_on=True,
        plot_again=True,
        num_cores=1,
        reduced_image_size_factor=6
    )
    # plot_original_image(
    #     data_manager,
    #     folder='reconstruction',
    #     axes_by=[],
    #     plot_by=['image', 'models', 'num_cells_per_dim', 'refinement'],
    #     axes_xy_proportions=(15, 15),
    #     numbers_on=True
    # )
