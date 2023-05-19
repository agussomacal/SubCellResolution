from functools import partial

import numpy as np

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.visualization import generic_plot
from experiments.subcell_paper.obera_experiments import fit_model, get_sub_cell_model, image_reconstruction, \
    plot_reconstruction, piecewise_constant
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.Curves.VanderCurves import CurveVandermondePolynomial, CurveVanderCircle


@fit_model
def linear(refinement: int, iterations: int, central_cell_extra_weight: float):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=1)), refinement,
        "LinearPoint", iterations, central_cell_extra_weight)


@fit_model
def linear_avg(refinement: int, iterations: int, central_cell_extra_weight: float):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=1)), refinement,
        "LinearAvg", iterations, central_cell_extra_weight)


@fit_model
def quadratic(refinement: int, iterations: int, central_cell_extra_weight: float):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=2)), refinement,
        "QuadraticPoint", iterations, central_cell_extra_weight)


@fit_model
def quadratic_avg(refinement: int, iterations: int, central_cell_extra_weight: float):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=2)), refinement,
        "QuadraticAvg", iterations, central_cell_extra_weight)


@fit_model
def circle(refinement: int, iterations: int, central_cell_extra_weight: float):
    return get_sub_cell_model(partial(ValuesCurveCellCreator, vander_curve=CurveVanderCircle), refinement,
                              "CirclePoint", iterations, central_cell_extra_weight)


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='AERO',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "models",
        piecewise_constant,
        linear,
        quadratic,
        circle,
        linear_avg,
        quadratic_avg
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
        save_on_iteration=1,
        refinement=[1],
        num_cells_per_dim=[14, 16, 20, 24, 30, 40, 56],  # 42 * 2
        # num_cells_per_dim=[20],  # 42 * 2
        noise=[0],
        image=[
            "Ellipsoid_1680x1680.png",
        ],
        iterations=[0],  # 500
        # iterations=[0],  # 500
        reconstruction_factor=[1],
        # reconstruction_factor=[6],
        # central_cell_extra_weight=[0, 100],
        central_cell_extra_weight=[0],
    )

    generic_plot(data_manager, x="N", y="mse", label="models",
                 # plot_func=NamedPartial(sns.lineplot, marker=".", linestyle="--"),
                 log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=lambda reconstruction_error: np.mean(reconstruction_error),
                 axes_by=["iterations", "central_cell_extra_weight"])

    generic_plot(data_manager, x="time", y="mse", label="models",
                 # plot_func=NamedPartial(sns.lineplot, marker=".", linestyle="--"),
                 log="xy", time=lambda time_to_fit: time_to_fit,
                 mse=lambda reconstruction_error: np.mean(reconstruction_error),
                 axes_by=["iterations", "central_cell_extra_weight"])

    plot_reconstruction(
        data_manager,
        name="Reconstruction",
        folder='reconstruction',
        axes_by=['models'],
        plot_by=['image', "num_cells_per_dim", 'refinement', "iterations", 'central_cell_extra_weight'],
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
