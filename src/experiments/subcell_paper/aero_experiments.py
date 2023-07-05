import time
from functools import partial

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline, FunctionBlock
from PerplexityLab.miscellaneous import NamedPartial
from PerplexityLab.visualization import generic_plot
from experiments.subcell_paper.function_families import calculate_averages_from_curve, calculate_averages_from_image
from experiments.subcell_paper.global_params import SUB_CELL_DISCRETIZATION2BOUND_ERROR, OBERA_ITERS, \
    CCExtraWeight
from experiments.subcell_paper.obera_experiments import get_sub_cell_model, get_shape, plot_reconstruction
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.TaylorCurveCellCreator import TaylorFromVanderCurveCellCreator, \
    TaylorCircleCurveCellCreator
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator
from lib.CellCreators.RegularCellCreator import MirrorCellCreator
from lib.CellIterators import iterate_all
from lib.CellOrientators import BaseOrientator, OrientPredefined
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.Curves.VanderCurves import CurveVanderCircle
from lib.SmoothnessCalculators import indifferent
from lib.StencilCreators import StencilCreatorFixedShape, StencilCreatorAdaptive
from lib.SubCellReconstruction import SubCellReconstruction, ReconstructionErrorMeasureBase, CellCreatorPipeline


def fit_model(sub_cell_model):
    def decorated_func(image, noise, metric, sub_discretization2bound_error):
        # image = load_image(image)
        # avg_values = calculate_averages_from_image(image, num_cells_per_dim)
        np.random.seed(42)
        avg_values = image + np.random.uniform(-noise, noise, size=image.shape)

        model = sub_cell_model(metric)

        t0 = time.time()
        model.fit(average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"))
        t_fit = time.time() - t0

        t0 = time.time()
        reconstruction = model.reconstruct_by_factor(resolution_factor=sub_discretization2bound_error)
        t_reconstruct = time.time() - t0
        # if refinement is set in place
        reconstruction = calculate_averages_from_image(reconstruction, tuple(
            (np.array(np.shape(image)) * sub_discretization2bound_error).tolist()))

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
def piecewise_constant(*args):
    return SubCellReconstruction(
        name="PiecewiseConstant",
        smoothness_calculator=indifferent,
        reconstruction_error_measure=ReconstructionErrorMeasureBase(),
        refinement=1,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=iterate_all,  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=MirrorCellCreator(dimensionality=2)
            )
        ]
    )


@fit_model
def elvira(metric):
    return get_sub_cell_model(ELVIRACurveCellCreator, 1, "ELVIRA", 0, 0, metric,
                              orientators=[OrientPredefined(predefined_axis=0), OrientPredefined(predefined_axis=1)])


@fit_model
def elvira_100(metric):
    return get_sub_cell_model(ELVIRACurveCellCreator, 1, "ELVIRA100", 0, CCExtraWeight, metric,
                              orientators=[OrientPredefined(predefined_axis=0), OrientPredefined(predefined_axis=1)])


@fit_model
def elvira_grad_oriented(metric):
    return get_sub_cell_model(ELVIRACurveCellCreator, 1, "ELVIRAGRAD", 0, CCExtraWeight, metric)


@fit_model
def elvira_go100_ref2(metric):
    return get_sub_cell_model(ELVIRACurveCellCreator, 2, "ELVIRAGRAD", 0, CCExtraWeight, metric)


@fit_model
def linear_obera(metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=100)), 1,
        "LinearOpt", OBERA_ITERS, CCExtraWeight, metric)


@fit_model
def linear_avg_100(metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=100)), 1,
        "LinearAvg100", 0, CCExtraWeight, metric)


@fit_model
def linear_avg(metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=0)), 1,
        "LinearAvg", 0, CCExtraWeight, metric)


@fit_model
def quadratic_obera_non_adaptive(metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticOptNonAdaptive", OBERA_ITERS, CCExtraWeight, metric)


@fit_model
def quadratic_obera(metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticOpt", OBERA_ITERS, CCExtraWeight, metric,
        StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3))


@fit_model
def quadratic_avg(metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticAvg", 0, CCExtraWeight, metric,
        StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3))


@fit_model
def quadratic_avg_ref2(metric):
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=2, ccew=CCExtraWeight)), 2,
        "QuadraticAvg", 0, CCExtraWeight, metric,
        StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3))


@fit_model
def circle_avg(metric):
    return get_sub_cell_model(
        partial(TaylorCircleCurveCellCreator, ccew=CCExtraWeight), 1,
        "CircleAvg", 0, CCExtraWeight, metric,
        StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3))


@fit_model
def circle_vander_avg(metric):
    return get_sub_cell_model(
        partial(TaylorFromVanderCurveCellCreator, curve=CurveVanderCircle, degree=2, ccew=CCExtraWeight), 1,
        "CircleAvg", 0, CCExtraWeight, metric,
        StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3))


# @fit_model
# def circle(iterations: int, metric):
#     return get_sub_cell_model(partial(ValuesCurveCellCreator, vander_curve=CurveVanderCircle), 1,
#                               "CirclePoint", iterations, central_cell_extra_weight, metric)


# generic_plot(data_manager, x="N", y="mse", label="models",
#              plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
#              log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
#              mse=lambda reconstruction, image4error: ((np.array(reconstruction) - image4error) ** 2).ravel()
#              )
# @plot_decorator(unlist=False, vars_not_to_include_in_name=["model_color", "ylim", "xlim"])
# def plot_convergence(self, ax, model, num_cells_per_dim, regular_errors, interface_errors, noise,
#                      model_color=None, log="", ylim=None, xlim=None, style=PLOT_STYLE_DEFAULT,
#                      indicator_hypothesis=False, **kwargs):
#     if model_color is None:
#         model_color = {mn: sns.color_palette("colorblind")[i] for i, mn in enumerate(sorted(list(set(model_name))))}
#     noise_alpha = np.sort(np.unique(noise))
#     noise_alpha_dict = dict(zip(noise_alpha, 1 - np.linspace(0, 0.5, len(noise_alpha))))
#     for mn, df in df2plot(grouping_var_name="model_name", sorting_var_name=["num_cells_per_dim"],
#                           model_name=model_name, num_cells_per_dim=num_cells_per_dim, noise=noise,
#                           errors=np.array(list(
#                               map(lambda rie: np.mean(np.append(*rie)), zip(regular_errors, interface_errors))))):
#         for eta, data in df.groupby("noise"):
#             if indicator_hypothesis:
#                 ax.plot(xlim, np.array(xlim)**(-1/2)*eta, c="gray", alpha=noise_alpha_dict[eta], linestyle=":")
#             else:
#                 ax.axhline(eta / 2, c="gray", alpha=noise_alpha_dict[eta], linestyle=":")
#
#             ax.text(x=np.min(num_cells_per_dim) * np.exp(-0.5), y=eta,
#                     s=r"$\eta={} \; 10^{{{}}}$".format(*format(eta, ".1E").replace("E-0", "E-").split("E")),
#                     fontsize='xx-small', horizontalalignment="left", verticalalignment='center',
#                     alpha=noise_alpha_dict[eta], fontstyle="italic")
#
#             # fit rate in 0 noise case
#             if eta == 0:
#                 rate, origin = np.ravel(np.linalg.lstsq(
#                     np.vstack([np.log(np.array(data.num_cells_per_dim)), np.ones(len(data))]).T,
#                     np.log(data.errors.values).reshape((-1, 1)), rcond=None)[0])
#                 ax.plot(data.num_cells_per_dim ** 2, data.num_cells_per_dim ** rate * np.exp(origin), "-",
#                         c=model_color[mn], label=f"{mn}: {rate:.2f}")
#
#             ax.plot(data.num_cells_per_dim ** 2, data.errors, "--", marker=".",
#                     alpha=noise_alpha_dict[eta], c=model_color[mn])
#
#     if 'x' in log:
#         ax.set_xscale('log')
#     if 'y' in log:
#         ax.set_yscale('log')
#
#     ax.set_xlabel(r"$n$")
#     ax.set_ylabel(r"$||u-\tilde u ||_{L_1}$")
#
#     if ylim is not None:
#         ax.set_ylim(ylim)
#     if xlim is not None:
#         ax.set_xlim(xlim)
#
#     set_style(style)

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
        # piecewise_constant,
        # elvira,
        # elvira_100,
        elvira_grad_oriented,
        # linear_obera,
        # linear_avg,
        linear_avg_100,
        # quadratic_obera_non_adaptive,
        # quadratic_obera,
        quadratic_avg,
        # elvira_go100_ref2,
        # quadratic_avg_ref2,
        circle_avg,
        circle_vander_avg,
        recalculate=False
    )
    num_cells_per_dim = np.logspace(np.log10(20), np.log10(100), num=10, dtype=int).tolist()
    lab.execute(
        data_manager,
        num_cores=15,
        forget=False,
        save_on_iteration=1,
        num_cells_per_dim=num_cells_per_dim,
        noise=[0],
        shape_name=[
            "Circle"
        ],
        sub_discretization2bound_error=[SUB_CELL_DISCRETIZATION2BOUND_ERROR],
        metric=[2]
    )

    generic_plot(data_manager,
                 name="Convergence",
                 x="N", y="mse", label="models", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=lambda reconstruction, image4error: np.mean(
                     ((np.array(reconstruction) - image4error) ** 2).ravel())
                 )

    generic_plot(data_manager,
                 name="TimeComplexity",
                 x="time", y="mse", label="models", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker=".", linestyle="--"),
                 log="xy", time=lambda time_to_fit: time_to_fit,
                 mse=lambda reconstruction, image4error: ((np.array(reconstruction) - image4error) ** 2).ravel()
                 )

    plot_reconstruction(
        data_manager,
        name="",
        folder='ReconstructionComparison',
        axes_by=['models'],
        plot_by=['num_cells_per_dim'],
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
        # trim=trim
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
