import time
from functools import partial

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline, FunctionBlock
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot
from experiments.subcell_paper.global_params import SUB_CELL_DISCRETIZATION2BOUND_ERROR, CCExtraWeight
from experiments.subcell_paper.obera_experiments import get_shape, get_sub_cell_model
from experiments.subcell_paper.tools import calculate_averages_from_curve
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE
from lib.CellCreators.CurveCellCreators.ParametersCurveCellCreators import DefaultPolynomialCurveCellCreator, \
    DefaultCircleCurveCellCreator
from lib.CellCreators.CurveCellCreators.TaylorCurveCellCreator import TaylorCircleCurveCellCreator
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesDefaultCircleCellCreator, \
    ValuesDefaultCurveCellCreator, ValuesDefaultLinearCellCreator, ValuesCircleCellCreator, ValuesCurveCellCreator, \
    ValuesLinearCellCreator
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.Curves.VanderCurves import CurveVandermondePolynomial


def fit_model(sub_cell_model):
    def decorated_func(image, image4error, noise, refinement, iterations, central_cell_extra_weight, metric,
                       sub_discretization2bound_error):
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
            "error": np.mean(np.abs(reconstruction - image4error)),
            "time_to_reconstruct": t_reconstruct
        }

    # need to change the name so the lab experiment saves the correct name and not the uniformly "decorated_func"
    # the other option is to pass to the block the name we wish to associate to the function.
    decorated_func.__name__ = sub_cell_model.__name__
    return decorated_func


# -------------------- Linear -------------------- #
@fit_model
def linear_aero_param(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=central_cell_extra_weight),
                natural_params=True),
        refinement, "Linear", iterations, central_cell_extra_weight, metric)


@fit_model
def linear_aero_reparam(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=central_cell_extra_weight),
                natural_params=False),
        refinement, "Linear", iterations, central_cell_extra_weight, metric)


@fit_model
def linear_points_param(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=1, ccew=central_cell_extra_weight),
                natural_params=True),
        refinement, "Linear", iterations, central_cell_extra_weight, metric)


@fit_model
def linear_points_reparam(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=1, ccew=central_cell_extra_weight),
                natural_params=False),
        refinement, "Linear", iterations, central_cell_extra_weight, metric)


@fit_model
def linear_dafault_param(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(DefaultPolynomialCurveCellCreator, degree=1), refinement,
        "Linear", iterations, central_cell_extra_weight, metric)


# -------------------- CIRCLE -------------------- #
@fit_model
def circle_aero_param(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(TaylorCircleCurveCellCreator, ccew=central_cell_extra_weight, natural_params=True), refinement,
        "Circle",
        iterations, central_cell_extra_weight, metric)


@fit_model
def circle_aero_reparam(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(TaylorCircleCurveCellCreator, ccew=central_cell_extra_weight, natural_params=False), refinement,
        "Circle",
        iterations, central_cell_extra_weight, metric)


@fit_model
def circle_i(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(partial(ValuesCircleCellCreator, natural_params=True), refinement, "Circle",
                              iterations, central_cell_extra_weight, metric)


@fit_model
def circle_pi(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(ValuesCircleCellCreator, refinement, "Circle",
                              iterations, central_cell_extra_weight, metric)


@fit_model
def circle_p(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(ValuesDefaultCircleCellCreator, refinement,
                              "Circle",
                              iterations, central_cell_extra_weight, metric)


@fit_model
def circle(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(DefaultCircleCurveCellCreator, refinement, "Circle",
                              iterations, central_cell_extra_weight, metric)


# -------------------- Quadratic -------------------- #
@fit_model
def quadratic_i(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=2),
                natural_params=True), refinement,
        "Quadratic", iterations, central_cell_extra_weight, metric)


@fit_model
def quadratic_pi(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=2)), refinement,
        "Quadratic", iterations, central_cell_extra_weight, metric)


@fit_model
def quadratic_p(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(ValuesDefaultCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=2)), refinement,
        "Quadratic", iterations, central_cell_extra_weight, metric)


@fit_model
def quadratic(refinement: int, iterations: int, central_cell_extra_weight: float, metric):
    return get_sub_cell_model(
        partial(DefaultPolynomialCurveCellCreator, degree=2), refinement,
        "Quadratic", iterations, central_cell_extra_weight, metric)


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='OBERAiterations',
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
        linear_dafault_param,
        linear_points_param,
        linear_points_reparam,
        linear_aero_param,
        linear_aero_reparam,
        # quadratic,
        # quadratic_p,
        # quadratic_pi,
        # quadratic_i,
        # circle,
        # circle_p,
        # circle_pi,
        # circle_i,
        # circle_aero,
        # circle_aero_reparam,
        recalculate=False
    )
    metrics = [1, 2]
    num_cores = 15
    lab.execute(
        data_manager,
        num_cores=num_cores,
        forget=False,
        save_on_iteration=None,
        refinement=[1],
        # num_cells_per_dim=[10, 14] + np.logspace(np.log10(20), np.log10(100), num=10, dtype=int).tolist()[:1],
        num_cells_per_dim=[20],
        noise=[0],
        shape_name=[
            "Circle"
        ],
        iterations=[0, 1, 2, 3, 4, 5, 10],
        central_cell_extra_weight=[100, 10, 1, 0],
        sub_discretization2bound_error=[SUB_CELL_DISCRETIZATION2BOUND_ERROR],
        metric=metrics
    )

    # ---------- Effect of re-parametrization and warm start --------- #
    order = [
        "linear_dafault_param",
        "linear_points_param",
        "linear_points_reparam",
        "linear_aero_param",
        "linear_aero_reparam",
        'quadratic',
        'quadratic_p',
        'quadratic_i',
        'quadratic_pi',
        'circle',
        'circle_p',
        'circle_i',
        'circle_pi',
        'circle_aero',
        'circle_aero_reparam',
    ]

    # def variant(models):
    #     if "_" in models:
    #         if "aero" in models:
    #             new_name = "AERO initialization"
    #             if "reparam" in models:
    #                 new_name += " re-parameterized"
    #         else:
    #             new_name = models.split("_")[1].replace("i", " and warm start")
    #             if "p" in new_name:
    #                 new_name = new_name.replace("p", "re-parameterized")
    #             else:
    #                 new_name = "normal params" + new_name
    #     else:
    #         new_name = "normal params"
    #     return new_name

    def variant(models):
        models = models.replace("_reparam", " re-parameterized")
        models = models.replace("_param", "")
        models = models.replace("_", " ")
        return models


    # generic_plot(data_manager,
    #              name="FevalsReParamWarmStartEffect",
    #              x="fevals", y="error", label="variant",
    #              plot_func=NamedPartial(sns.lineplot,
    #                                     marker="o", linestyle="--",
    #                                     # hue_order=["normal params", "re-parameterized",
    #                                     #            "normal params and warm start",
    #                                     #            "re-parameterized and warm start"]
    #                                     ),
    #              log="y",
    #              time=lambda model: np.mean(np.array(list(model.times[CURVE_CELL_TYPE].values()))),
    #              fevals=lambda model: np.array(list(model.obera_fevals[CURVE_CELL_TYPE].values())),
    #              curve=lambda models: models.split("_")[0],
    #              variant=variant,
    #              sort_by=["models"],
    #              plot_by=["metric", "curve"],
    #              axes_by=["central_cell_extra_weight", ],
    #              )

    generic_plot(data_manager,
                 name="TimeReParamWarmStartEffect",
                 x="time", y="error", label="variant",
                 plot_func=NamedPartial(sns.lineplot,
                                        marker="o", linestyle="--",
                                        # hue_order=["normal params", "re-parameterized",
                                        #            "normal params and warm start",
                                        #            "re-parameterized and warm start"]
                                        ),
                 log="y",
                 time=lambda model: np.mean(np.array(list(model.times[CURVE_CELL_TYPE].values()))),
                 fevals=lambda model: np.array(list(model.obera_fevals[CURVE_CELL_TYPE].values())),
                 curve=lambda models: models.split("_")[0],
                 variant=variant,
                 sort_by=["models"],
                 plot_by=["metric", "curve"],
                 axes_by=["central_cell_extra_weight", ],
                 )

    generic_plot(data_manager,
                 name="ItersReParamWarmStartEffect",
                 x="iterations", y="error", label="variant",
                 plot_func=NamedPartial(sns.lineplot,
                                        marker="o", linestyle="--",
                                        # hue_order=["normal params", "re-parameterized",
                                        #            "normal params and warm start",
                                        #            "re-parameterized and warm start"]
                                        ),
                 log="y",
                 curve=lambda models: models.split("_")[0],
                 variant=variant,
                 sort_by=["models"],
                 plot_by=["metric", "curve"],
                 axes_by=["central_cell_extra_weight", ],
                 )
