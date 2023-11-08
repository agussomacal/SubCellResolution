import time
from functools import partial

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline, FunctionBlock
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import perplex_plot, generic_plot, one_line_iterator
from experiments.VizReconstructionUtils import plot_cells, plot_cells_identity, plot_cells_vh_classification_core, \
    plot_cells_not_regular_classification_core, plot_curve_core, draw_cell_borders
from experiments.subcell_paper.ex_aero import get_sub_cell_model, get_shape
from experiments.subcell_paper.global_params import SUB_CELL_DISCRETIZATION2BOUND_ERROR
from experiments.subcell_paper.tools import calculate_averages_from_curve
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE
from lib.CellCreators.CurveCellCreators.ParametersCurveCellCreators import DefaultCircleCurveCellCreator, \
    DefaultPolynomialCurveCellCreator
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator, \
    ValuesDefaultCurveCellCreator, ValuesLinearCellCreator, ValuesDefaultLinearCellCreator, \
    ValuesDefaultCircleCellCreator, ValuesCircleCellCreator
from lib.Curves.VanderCurves import CurveVandermondePolynomial


def fit_model(sub_cell_model):
    def decorated_func(image, noise, refinement, iterations, central_cell_extra_weight, metric,
                       sub_discretization2bound_error):
        # image = load_image(image)
        # avg_values = calculate_averages_from_image(image, num_cells_per_dim)
        np.random.seed(42)
        avg_values = image + np.random.uniform(-noise, noise, size=image.shape)

        model = sub_cell_model(refinement, iterations, central_cell_extra_weight, metric)

        t0 = time.time()
        model.fit(average_values=avg_values,
                  indexer=ArrayIndexerNd(avg_values, "cyclic"))
        t_fit = time.time() - t0

        t0 = time.time()
        reconstruction = model.reconstruct_by_factor(resolution_factor=sub_discretization2bound_error)
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


@fit_model
def linear_i(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(ValuesLinearCellCreator, natural_params=True), refinement,
        "Linear", iterations, central_cell_extra_weight, metric)


@fit_model
def quadratic_i(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=2),
                natural_params=True), refinement,
        "Quadratic", iterations, central_cell_extra_weight, metric)


@fit_model
def circle_i(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(partial(ValuesCircleCellCreator, natural_params=True), refinement, "Circle",
                              iterations, central_cell_extra_weight, metric)


@fit_model
def linear_pi(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        ValuesLinearCellCreator, refinement,
        "Linear", iterations, central_cell_extra_weight, metric)


@fit_model
def quadratic_pi(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=2)), refinement,
        "Quadratic", iterations, central_cell_extra_weight, metric)


@fit_model
def circle_pi(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(ValuesCircleCellCreator, refinement, "Circle",
                              iterations, central_cell_extra_weight, metric)


@fit_model
def linear_p(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(ValuesDefaultLinearCellCreator, refinement, "Linear", iterations,
                              central_cell_extra_weight, metric)


@fit_model
def quadratic_p(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(ValuesDefaultCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=2)), refinement,
        "Quadratic", iterations, central_cell_extra_weight, metric)


@fit_model
def circle_p(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(ValuesDefaultCircleCellCreator, refinement,
                              "Circle",
                              iterations, central_cell_extra_weight, metric)


@fit_model
def linear(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(DefaultPolynomialCurveCellCreator, degree=1), refinement,
        "Linear", iterations, central_cell_extra_weight, metric)


@fit_model
def quadratic(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(DefaultPolynomialCurveCellCreator, degree=2), refinement,
        "Quadratic", iterations, central_cell_extra_weight, metric)


@fit_model
def circle(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(DefaultCircleCurveCellCreator, refinement, "Circle",
                              iterations, central_cell_extra_weight, metric)


@perplex_plot()
@one_line_iterator
def plot_reconstruction(fig, ax, image4error, num_cells_per_dim, model, reconstruction, alpha=0.5,
                        plot_original_image=True,
                        difference=False, plot_curve=True, plot_curve_winner=False, plot_vh_classification=True,
                        plot_singular_cells=True, cmap="magma", trim=((0, 1), (0, 1)), numbers_on=True):
    """

    :param fig:
    :param ax:
    :param image4error:
    :param num_cells_per_dim:
    :param model:
    :param reconstruction:
    :param alpha:
    :param plot_original_image:
    :param difference:
    :param plot_curve:
    :param plot_curve_winner:
    :param plot_vh_classification:
    :param plot_singular_cells:
    :param cmap:
    :param trim: (xlims, ylims)
    :param numbers_on:
    :return:
    """
    model_resolution = np.array(model.resolution)

    if plot_original_image:
        plot_cells(ax, colors=image4error, mesh_shape=model_resolution, alpha=alpha, cmap="Greys_r",
                   vmin=np.min(image4error), vmax=np.max(image4error))
    if difference:
        plot_cells(ax, colors=reconstruction - image4error, mesh_shape=model_resolution, alpha=alpha, cmap=cmap,
                   vmin=-1, vmax=1)
    else:
        plot_cells(ax, colors=reconstruction, mesh_shape=model_resolution, alpha=alpha, cmap=cmap, vmin=-1, vmax=1)

    if plot_curve:
        if plot_curve_winner:
            plot_cells_identity(ax, model_resolution, model.cells, alpha=0.8)
            # plot_cells_type_of_curve_core(ax, model.resolution, model.cells, alpha=0.8)
        elif plot_vh_classification:
            plot_cells_vh_classification_core(ax, model_resolution, model.cells, alpha=0.8)
        elif plot_singular_cells:
            plot_cells_not_regular_classification_core(ax, model_resolution, model.cells, alpha=0.8)
        plot_curve_core(ax, curve_cells=[cell for cell in model.cells.values() if
                                         cell.CELL_TYPE == CURVE_CELL_TYPE])

    draw_cell_borders(
        ax, mesh_shape=num_cells_per_dim,
        refinement=model_resolution // num_cells_per_dim,
        numbers_on=numbers_on,
        prop_ticks=10 / num_cells_per_dim  # each 10 cells a tick
    )

    ax.set_xlim((-0.5 + trim[0][0] * model.resolution[0], model.resolution[0] * trim[0][1] - 0.5))
    ax.set_ylim((model.resolution[1] * trim[1][0] - 0.5, model.resolution[1] * trim[1][1] - 0.5))


def get_reconstructed_subcells_coords(coord, sub_discretization2bound_error, reconstruction):
    return reconstruction[list(
        map(lambda i: np.arange(i * sub_discretization2bound_error, (i + 1) * sub_discretization2bound_error),
            coord))]


def singular_error(reconstruction, image4error, model, sub_discretization2bound_error, num_cells_per_dim):
    return np.array(list(map(np.mean,
                             map(partial(get_reconstructed_subcells_coords,
                                         reconstruction=np.abs(np.array(reconstruction) - image4error),
                                         sub_discretization2bound_error=sub_discretization2bound_error),
                                 model.obera_fevals[CURVE_CELL_TYPE].keys()
                                 )
                             )
                         )) / num_cells_per_dim ** 2


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='OBERA',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "precompute_images",
        FunctionBlock(
            "getimages",
            lambda shape_name, num_cells_per_dim: {
                "image": calculate_averages_from_curve(
                    get_shape(shape_name),
                    (num_cells_per_dim,
                     num_cells_per_dim))}
        )
    )

    lab.define_new_block_of_functions(
        "precompute_error_resolution",
        FunctionBlock(
            "subresolution",
            lambda shape_name, num_cells_per_dim, sub_discretization2bound_error: {
                "image4error": calculate_averages_from_curve(
                    get_shape(shape_name),
                    (num_cells_per_dim * sub_discretization2bound_error,
                     num_cells_per_dim * sub_discretization2bound_error))}
        )
    )

    lab.define_new_block_of_functions(
        "models",
        linear,
        quadratic,
        circle,
        linear_p,
        quadratic_p,
        circle_p,
        linear_pi,
        quadratic_pi,
        circle_pi,
        linear_i,
        quadratic_i,
        circle_i,
        recalculate=False
    )
    metrics = [2]
    num_cores = 15
    lab.execute(
        data_manager,
        num_cores=num_cores,
        forget=False,
        save_on_iteration=num_cores,
        refinement=[1],
        # num_cells_per_dim=[10, 14] + np.logspace(np.log10(20), np.log10(100), num=10, dtype=int).tolist()[:1],
        num_cells_per_dim=[20],
        noise=[0],
        shape_name=[
            "Circle"
        ],
        iterations=[500],  # 500
        central_cell_extra_weight=[0, 100],
        sub_discretization2bound_error=[SUB_CELL_DISCRETIZATION2BOUND_ERROR],
        metric=metrics
    )
    metrics = [1, 1.5, 2, 4]
    lab.define_new_block_of_functions(
        "models",
        linear_pi,
        quadratic_pi,
        circle_pi,
        recalculate=False
    )
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=1,
        refinement=[1],
        # num_cells_per_dim=[10, 14] + np.logspace(np.log10(20), np.log10(100), num=10, dtype=int).tolist()[:1],
        num_cells_per_dim=[20],
        noise=[0],
        shape_name=[
            "Circle"
        ],
        iterations=[500],  # 500
        central_cell_extra_weight=[100],
        sub_discretization2bound_error=[SUB_CELL_DISCRETIZATION2BOUND_ERROR],
        metric=metrics
    )

    trim = ((4 / 20, 11 / 20), (5 / 20, 11 / 20))
    metric = 2
    plot_reconstruction(
        data_manager,
        name="ReconstructionComparison",
        folder='ReconstructionComparison',
        num_cells_per_dim=20,
        metric=2,
        models=["linear", "quadratic", "circle"],
        plot_by=['models', 'central_cell_extra_weight'],
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
        trim=trim
    )

    # ---------- Effect of re-parametrization and warm start --------- #
    order = [
        'linear',
        'linear_p',
        'linear_i',
        'linear_pi',
        'quadratic',
        'quadratic_p',
        'quadratic_i',
        'quadratic_pi',
        'circle',
        'circle_p',
        'circle_i',
        'circle_pi',
    ]


    def variant(models):
        if "_" in models:
            new_name = models.split("_")[1].replace("i", " and warm start")
            if "p" in new_name:
                new_name = new_name.replace("p", "re-parameterized")
            else:
                new_name = "normal params" + new_name
        else:
            new_name = "normal params"
        print(new_name)
        return new_name


    generic_plot(data_manager,
                 name="ReParamWarmStartEffect",
                 x="curve", y="fevals", label="variant",
                 plot_func=NamedPartial(sns.barplot, order=["linear", "quadratic", "circle"],
                                        hue_order=["normal params", "re-parameterized",
                                                   "normal params and warm start",
                                                   "re-parameterized and warm start"]),
                 log="",
                 curve=lambda models: models.split("_")[0],
                 variant=variant,
                 sort_by=["models"],
                 time=lambda model: np.mean(np.array(list(model.times[CURVE_CELL_TYPE].values()))),
                 fevals=lambda model: np.array(list(model.obera_fevals[CURVE_CELL_TYPE].values())),
                 metric=2,
                 ylim=(0, 200),
                 plot_by=["central_cell_extra_weight"])

    # ---------- Effect of central_cell_extra_weight metric --------- #
    generic_plot(data_manager,
                 name="CCWeightEffect",
                 x="curve", y="error", label="central_cell_extra_weight",
                 plot_func=NamedPartial(sns.barplot, order=["linear", "quadratic", "circle"], hue_order=[0, 100]),
                 log="y",
                 curve=lambda models: models.split("_")[0],
                 sort_by=["models"],
                 metric=2,
                 ylim=(1e-9, 1e-3),
                 models=["linear_pi", "quadratic_pi", "circle_pi"],
                 error=lambda reconstruction, image4error, model, sub_discretization2bound_error,
                              num_cells_per_dim:
                 singular_error(reconstruction, image4error, model, sub_discretization2bound_error,
                                num_cells_per_dim),
                 # axes_by=["metric"],
                 )

    # ---------- Effect of p metric --------- #
    generic_plot(data_manager,
                 name="pmetricEffect_fevels",
                 x="curve", y="fevals", label="metric",
                 plot_func=NamedPartial(sns.barplot, order=["linear", "quadratic", "circle"], hue_order=metrics),
                 log="",
                 curve=lambda models: models.split("_")[0],
                 sort_by=["models"],
                 models=["linear_pi", "quadratic_pi", "circle_pi"],
                 fevals=lambda model: np.array(list(model.obera_fevals[CURVE_CELL_TYPE].values())),
                 central_cell_extra_weight=100,
                 axes_by=[],
                 plot_by=["central_cell_extra_weight"]
                 )

    generic_plot(data_manager,
                 name="pmetricEffect_error",
                 x="curve", y="error", label="metric",
                 plot_func=NamedPartial(sns.barplot, order=["linear", "quadratic", "circle"], hue_order=metrics),
                 log="y",
                 curve=lambda models: models.split("_")[0],
                 sort_by=["models"],
                 models=["linear_pi", "quadratic_pi", "circle_pi"],
                 error=lambda reconstruction, image4error, model, sub_discretization2bound_error,
                              num_cells_per_dim:
                 singular_error(reconstruction, image4error, model, sub_discretization2bound_error,
                                num_cells_per_dim),
                 central_cell_extra_weight=100,
                 axes_by=[],
                 plot_by=["central_cell_extra_weight"])

    plot_reconstruction(
        data_manager,
        name="CompareModels",
        folder='CompareModels',
        # num_cells_per_dim=20,
        metric=metric,
        # models=["linear", "quadratic", "circle"],
        # models=["linear_pi", "quadratic_pi", "circle_pi"],
        axes_by=['models'],
        plot_by=['central_cell_extra_weight', "metric"],
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
        trim=trim
    )

    plot_reconstruction(
        data_manager,
        name="Reconstruction",
        folder='reconstruction',
        num_cells_per_dim=20,
        metric=metrics,
        models=["linear", "quadratic", "circle"],
        axes_by=['metric'],
        plot_by=['central_cell_extra_weight', "models"],
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
        trim=trim
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
