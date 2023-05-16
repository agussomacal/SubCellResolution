import time

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.visualization import generic_plot
from experiments.image_reconstruction import plot_reconstruction
from experiments.subcell_paper.function_families import load_image, calculate_averages_from_image
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE
from lib.CellCreators.RegularCellCreator import MirrorCellCreator, \
    PolynomialRegularCellCreator
from lib.CellIterators import iterate_all
from lib.CellOrientators import BaseOrientator
from lib.SmoothnessCalculators import indifferent
from lib.StencilCreators import StencilCreatorFixedShape
from lib.SubCellReconstruction import SubCellReconstruction, CellCreatorPipeline, ReconstructionErrorMeasureBase


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
def fixed_polynomial_degree2(refinement: int):
    return get_sub_cell_model(PolynomialRegularCellCreator(degree=2, noisy=False, weight_function=None,
                                                           dimensionality=2, full_rank=True),
                              StencilCreatorFixedShape(stencil_shape=(3, 3)),
                              refinement, "FixedPolynomialDegree2")


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
        name='WENO',
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
        piecewise_constant,
        fixed_polynomial_degree2,
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
        amplitude=[0, 1e-2, 2e-2, 5e-2, 1e-1],
        refinement=[1],
        # num_cells_per_dim=[14, 20, 28, 42, 42 * 2],  # 42 * 2
        num_cells_per_dim=[20],  # 42 * 2
        noise=[0],
        image=[
            "Ellipsoid_1680x1680.png",
        ],
        # reconstruction_factor=[1],
        reconstruction_factor=[6],
    )

    generic_plot(data_manager, x="amplitude", y="mse", label="models",
                 # plot_func=NamedPartial(sns.lineplot, marker=".", linestyle="--"),
                 log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=lambda reconstruction_error: np.mean(reconstruction_error),
                 axes_by=["num_cells_per_dim"])

    generic_plot(data_manager, x="time", y="mse", label="models",
                 # plot_func=NamedPartial(sns.lineplot, marker=".", linestyle="--"),
                 log="xy", time=lambda time_to_fit: time_to_fit,
                 mse=lambda reconstruction_error: np.mean(reconstruction_error),
                 axes_by=["num_cells_per_dim"])

    plot_reconstruction(
        data_manager,
        name="Reconstruction",
        folder='reconstruction',
        axes_by=['models'],
        plot_by=['image', "num_cells_per_dim", 'refinement'],
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
