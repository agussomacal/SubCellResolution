import operator
import time
from collections import OrderedDict, namedtuple
from functools import partial
from itertools import chain

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot, make_data_frames, perplex_plot, one_line_iterator
from experiments.MLTraining.ml_cell_averages import kernel_circles_ml_model_points, kernel_lines_ml_model, \
    kernel_quadratics_ml_model, kernel_quadratics_points_ml_model
from experiments.MLTraining.ml_curve_params import lines_ml_model, quadratics7_points_ml_model, \
    quadratics7_params_ml_model, quadratics_ml_model
from experiments.MLTraining.ml_global_params import num_cores
from experiments.VizReconstructionUtils import plot_cells, plot_cells_identity, plot_cells_vh_classification_core, \
    plot_cells_not_regular_classification_core, plot_curve_core, draw_cell_borders
from experiments.subcell_paper.global_params import SUB_CELL_DISCRETIZATION2BOUND_ERROR, OBERA_ITERS, \
    CCExtraWeight, runsinfo, cblue, corange, cgreen, cred, cpurple, cbrown, cpink, cgray, cyellow, ccyan
from experiments.subcell_paper.tools import calculate_averages_from_curve, \
    curve_cells_fitting_times, singular_cells_mask, make_image_high_resolution
from lib.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CellCreatorBase import CURVE_CELL_TYPE
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.LearningCurveCellCreator import LearningCurveCellCreator
from lib.CellCreators.CurveCellCreators.RegularCellsSearchers import get_opposite_regular_cells_by_minmax
from lib.CellCreators.CurveCellCreators.TaylorCurveCellCreator import TaylorCircleCurveCellCreator
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator, \
    ValuesLineConsistentCurveCellCreator, ValuesCircleCellCreator
from lib.CellCreators.RegularCellCreator import MirrorCellCreator, PiecewiseConstantRegularCellCreator
from lib.CellIterators import iterate_all, iterate_by_condition_on_smoothness
from lib.CellOrientators import BaseOrientator, OrientByGradient
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.Curves.CurveCircle import CurveCircle, CircleParams
from lib.Curves.VanderCurves import CurveVandermondePolynomial
from lib.SmoothnessCalculators import indifferent, naive_piece_wise
from lib.StencilCreators import StencilCreatorFixedShape, StencilCreatorAdaptive
from lib.SubCellReconstruction import SubCellReconstruction, ReconstructionErrorMeasureBase, CellCreatorPipeline, \
    ReconstructionErrorMeasureML, reconstruct_by_factor, ReconstructionErrorMeasure

# ========== =========== ========== =========== #
#               Models for paper                #
# ========== =========== ========== =========== #
PlotStyle = namedtuple("PlotStyle", "color marker linestyle", defaults=["black", "o", "--"])

accepted_models = {
    "LinearModels": OrderedDict([
        ("elvira", PlotStyle(color=cred)),
        ("elvira_w_oriented", PlotStyle(color=corange)),
        ("elvira_w_oriented_l1", PlotStyle(color=cbrown)),
        ("linear_obera", PlotStyle(color=cblue)),
        ("linear_obera_w", PlotStyle(color=cpurple)),
        ("linear_obera_w_l2", PlotStyle(color=ccyan)),
        ("linear_aero", PlotStyle(color=cgray)),
        ("linear_aero_w", PlotStyle(color=cgreen)),
        ("linear_aero_consistent", PlotStyle(color=cpink)),
    ]),
    "HighOrderModels": OrderedDict([
        ("piecewise_constant", PlotStyle(color=cpink)),

        ("elvira_w_oriented", PlotStyle(color=corange)),
        ("linear_obera_w", PlotStyle(color=corange, marker=".", linestyle=":")),

        # ("quadratic_obera_non_adaptive_l1", PlotStyle(color=cbrown)),
        ("quadratic_aero", PlotStyle(color=cgreen)),
        # ("quadratic_obera_non_adaptive", PlotStyle(color=cred)),
        ("quadratic_obera_non_adaptive", PlotStyle(color=cgreen, marker=".", linestyle=":")),
        # ("quadratic_obera", PlotStyle(color=cred, marker=".", linestyle=":")),
        # ("quadratic_obera_l1", PlotStyle(color=cgray, marker=".", linestyle=":")),

        ("cubic_aero", PlotStyle(color=cblue)),
        # ("cubic_obera", PlotStyle(color=cblue, marker=".", linestyle=":")),

        ("quartic_aero", PlotStyle(color=cpurple)),
        # ("quartic_obera", PlotStyle(color=cpurple, marker=".", linestyle=":")),

        # ("obera_circle", PlotStyle(color=cbrown)),
        ("obera_circle_vander", PlotStyle(color=cyellow)),
    ]),
    "MLLinear": OrderedDict([
        ("elvira_w_oriented", PlotStyle(color=corange)),
        ("elvira_w_oriented_ml", PlotStyle(color=cred)),
        ("nn_linear", PlotStyle(color=cgreen)),
        ("linear_obera_w_ml", PlotStyle(color=cblue)),
        ("linear_obera_w", PlotStyle(color=cpurple)),
    ]),
    "MLQuadratic": OrderedDict([
        ("quadratic_obera_non_adaptive", PlotStyle(color=cred)),

        ("quadratic_obera_ml_params", PlotStyle(color=corange)),
        ("quadratic_obera_ml_points_adapt", PlotStyle(color=cpink)),
        ("quadratic_obera_ml_points", PlotStyle(color=ccyan)),

        ("quadratic_aero", PlotStyle(color=cgreen)),

        ("nn_quadratic_3x3", PlotStyle(color=cgray)),
        ("nn_quadratic_3x7", PlotStyle(color=cyellow)),
        ("nn_quadratic_3x7_params", PlotStyle(color=cblue)),
        ("nn_quadratic_3x7_adapt", PlotStyle(color=cbrown)),
        ("nn_quadratic_3x7_params_adapt", PlotStyle(color=cpurple)),
    ])
}

names_dict = {
    "piecewise_constant": "Piecewise Constant",

    "nn_linear": "NN Linear",
    "nn_quadratic_3x3": "NN Quadratic 3x3",
    "nn_quadratic_3x7": "NN Quadratic 3x7",
    "nn_quadratic_3x7_params": "NN Quadratic 3x7 params",
    "nn_quadratic_3x7_adapt": "NN Quadratic 3x7 Adapt S",
    "nn_quadratic_3x7_params_adapt": "NN Quadratic 3x7 params Adapt S",

    "elvira": "ELVIRA",
    "elvira_w": "ELVIRA-W",
    "elvira_w_oriented": "ELVIRA-W Oriented l2",
    "elvira_w_oriented_l1": "ELVIRA-W Oriented l1",
    "elvira_w_oriented_ml": "ELVIRA-W Oriented ML-ker l2",

    "linear_obera": "OBERA Linear l1",
    "linear_obera_w": "OBERA-W Linear l1",
    "linear_obera_w_l2": "OBERA-W Linear l2",
    "linear_aero": "AEROS Linear",
    "linear_aero_w": "AEROS-W Linear",
    "linear_aero_consistent": "AEROS Linear Column Consistent",
    "linear_obera_w_ml": "OBERA-W Linear ML-ker l1",

    "quadratic_obera_non_adaptive": "OBERA Quadratic 3x3 l2",
    "quadratic_obera_non_adaptive_l1": "OBERA Quadratic 3x3 l1",
    "quadratic_obera": "OBERA Quadratic l2",
    "quadratic_obera_l1": "OBERA Quadratic l1",
    "quadratic_obera_ml": "OBERA Quadratic ML-ker",
    "quadratic_aero": "AEROS Quadratic",
    "quadratic_obera_ml_params": "OBERA Quadratic ML-ker 3x3",
    "quadratic_obera_ml_points_adapt": "OBERA Quadratic ML-ker 3x3 ReParam Adapt S",
    "quadratic_obera_ml_points": "OBERA Quadratic ML-ker 3x3 ReParam",

    "cubic_aero": "AEROS Cubic",
    "cubic_obera": "OBERA Cubic",

    "quartic_aero": "AEROS Quartic",
    "quartic_obera": "OBERA Cubic",

    "obera_circle": "OBERA Circle",
    "obera_circle_vander": "OBERA Circle ReParam",
    "obera_circle_vander_ml": "OBERA Circle ReParam ML-ker",
    # "elvira_go100_ref2",
    # "quadratic_aero_ref2",
}

rateonly = list(filter(lambda x: "circle" not in x, names_dict.keys()))
runsinfo.append_info(
    **{k.replace("_", "-"): v for k, v in names_dict.items()}
)

# ========== =========== ========== =========== #
#            Experiments definition             #
# ========== =========== ========== =========== #
runsinfo.append_info(
    circler=0.232,
    circlex=0.511,
    circley=0.486
)


def get_shape(shape_name):
    if shape_name == "Circle":
        return CurveCircle(
            params=CircleParams(x0=runsinfo["circlex"], y0=runsinfo["circley"], radius=runsinfo["circler"]))
    else:
        raise Exception("Not implemented.")


def obtain_images(shape_name, num_cells_per_dim):
    image = calculate_averages_from_curve(get_shape(shape_name), (num_cells_per_dim, num_cells_per_dim))
    return {
        "image": image
    }


def obtain_image4error(shape_name, num_cells_per_dim, sub_discretization2bound_error, image):
    edge_mask = make_image_high_resolution(singular_cells_mask(image),
                                           reconstruction_factor=sub_discretization2bound_error)
    cells2reconstruct = list(zip(*np.where(edge_mask)))
    true_reconstruction = make_image_high_resolution(image, reconstruction_factor=sub_discretization2bound_error)
    true_reconstruction[edge_mask] = calculate_averages_from_curve(
        get_shape(shape_name),
        (num_cells_per_dim * sub_discretization2bound_error,
         num_cells_per_dim * sub_discretization2bound_error),
        cells2reconstruct=cells2reconstruct)[edge_mask]

    return {
        "image4error": true_reconstruction
    }


error_l1 = lambda reconstruction, image4error: np.mean(np.abs(reconstruction - image4error).ravel()) \
    if reconstruction is not None else np.nan

error_linf = lambda reconstruction, image4error: np.max(np.abs(reconstruction - image4error).ravel()) \
    if reconstruction is not None else np.nan


def efficient_reconstruction(model, avg_values, sub_discretization2bound_error):
    """
    Only reconstructs fully in the cells where there is discontinuity otherwise copies avgcells values
    :return:
    """

    edge_mask = singular_cells_mask(avg_values)
    edge_mask_hr = make_image_high_resolution(edge_mask,
                                              reconstruction_factor=sub_discretization2bound_error)
    cells2reconstruct = list(zip(*np.where(edge_mask)))
    t0 = time.time()
    reconstruction = make_image_high_resolution(avg_values, reconstruction_factor=sub_discretization2bound_error)
    reconstruction[edge_mask_hr] = \
        reconstruct_by_factor(cells=model.cells, resolution=model.resolution, cells2reconstruct=cells2reconstruct,
                              resolution_factor=sub_discretization2bound_error)[edge_mask_hr]
    t_reconstruct = time.time() - t0
    return reconstruction, t_reconstruct


def fit_model(sub_cell_model):
    def decorated_func(image, image4error, noise, sub_discretization2bound_error):
        np.random.seed(42)
        avg_values = image + np.random.uniform(-noise, noise, size=image.shape)

        model = sub_cell_model()
        print(f"Doing model: {decorated_func.__name__}")

        t0 = time.time()
        model.fit(average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"))
        t_fit = time.time() - t0

        reconstruction, t_reconstruct = efficient_reconstruction(model, avg_values, sub_discretization2bound_error)

        return {
            "model": model,
            "time_to_fit": t_fit,
            "error_l1": error_l1(reconstruction, image4error),
            "error_linf": error_linf(reconstruction, image4error),
            "time_to_reconstruct": t_reconstruct
        }

    # need to change the name so the lab experiment saves the correct name and not the uniformly "decorated_func"
    # the other option is to pass to the block the name we wish to associate to the function.
    decorated_func.__name__ = sub_cell_model.__name__
    return decorated_func


# ========== =========== ========== =========== #
#               Plots definition                #
# ========== =========== ========== =========== #

@perplex_plot(legend=False)
@one_line_iterator
def plot_reconstruction(fig, ax, image, image4error, num_cells_per_dim, model, sub_discretization2bound_error,
                        alpha=0.5, plot_original_image=True, difference=False, plot_curve=True, plot_curve_winner=False,
                        plot_vh_classification=True, plot_singular_cells=True, cmap="magma", trim=((0, 1), (0, 1)),
                        numbers_on=True):
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
    reconstruction, t_reconstruct = efficient_reconstruction(model, image, sub_discretization2bound_error)

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


@perplex_plot(group_by="models")
def plot_h_convergence(fig, ax, num_cells_per_dim, models, error_l1, error_linf, error="l1",
                       threshold=1.0 / np.sqrt(1000), rateonly=None, model_style=None, names_dict=None, *args,
                       **kwargs):
    error = np.array(error_l1 if error == "l1" else error_linf)
    name = f"{names_dict[str(models)]}"
    h = 1.0 / np.array(num_cells_per_dim)
    if rateonly is None or models in rateonly:
        valid_ix = h < threshold
        rate, origin = np.ravel(np.linalg.lstsq(
            np.vstack([np.log(h[valid_ix]), np.ones(np.sum(valid_ix))]).T,
            np.log(error[valid_ix]).reshape((-1, 1)), rcond=None)[0])
        # ax.plot(df[x].values[valid_ix], np.sqrt(df[x].values[valid_ix]) ** rate * np.exp(origin), "-",
        #         c=model_color[method], linewidth=3, alpha=0.5)
        name = name + f": O({abs(rate):.1f})"
    order = np.argsort(h)
    sns.lineplot(
        x=h[order], y=error[order], label=name, ax=ax, alpha=1,
        color=model_style[models].color if model_style is not None else None,
        marker=model_style[models].marker if model_style is not None else None,
        linestyle=model_style[models].linestyle if model_style is not None else None
    )
    # ax.plot(df[x], df[y], marker=".", linestyle=":", c=model_color[method], label=name)
    ax.set_xlabel(fr"$h$")
    ax.set_ylabel(r"$||u-\tilde u ||_{L^1}$")
    ax.set_xscale("log")
    ax.set_yscale("log")


def plot_convergence(data, x, y, hue, ax, threshold=np.sqrt(1000), rateonly=None, model_style=None, names_dict=None,
                     *args, **kwargs):
    # sns.scatterplot(data=data, x=x, y=y, hue=label, ax=ax)
    for method, df in data.groupby(hue, sort=False):
        name = f"{names_dict[str(method)]}"
        if rateonly is None or method in rateonly:
            hinv = np.sqrt(df[x].values)
            valid_ix = hinv > threshold
            rate, origin = np.ravel(np.linalg.lstsq(
                np.vstack([np.log(hinv[valid_ix]), np.ones(np.sum(valid_ix))]).T,
                np.log(df[y].values[valid_ix]).reshape((-1, 1)), rcond=None)[0])
            # ax.plot(df[x].values[valid_ix], np.sqrt(df[x].values[valid_ix]) ** rate * np.exp(origin), "-",
            #         c=model_color[method], linewidth=3, alpha=0.5)
            name = name + f": O({abs(rate):.1f})"
        sns.lineplot(
            x=df[x], y=df[y], label=name, ax=ax, alpha=1,
            color=model_style[method].color if model_style is not None else None,
            marker=model_style[method].marker if model_style is not None else None,
            linestyle=model_style[method].linestyle if model_style is not None else None,
        )
        # ax.plot(df[x], df[y], marker=".", linestyle=":", c=model_color[method], label=name)
    ax.set_xlabel(fr"${{{x}}}$")
    ax.set_ylabel(r"$||u-\tilde u ||_{L^1}$")

    # n = np.sort(np.unique(data[x]))
    # ax.plot(n, 1 / n, ":", c=cgray, linewidth=2, alpha=0.5, label=r"$O(h^{-1})$")
    # ax.plot(n, 1 / n ** 2, ":", c=cgray, linewidth=2, alpha=0.5, label=r"$O(h^{-2})$")
    # ax.plot(n, 1 / n ** 3, ":", c=cgray, linewidth=2, alpha=0.5, label=r"$O(h^{-3})$")


# ========== =========== ========== =========== #
#               Models definition               #
# ========== =========== ========== =========== #

def get_sub_cell_model(curve_cell_creator, refinement, name, iterations, central_cell_extra_weight, metric,
                       stencil_creator=StencilCreatorFixedShape((3, 3)),
                       orientator=OrientByGradient(kernel_size=(5, 5), dimensionality=2, angle_threshold=45),
                       reconstruction_error_measure=None):
    return SubCellReconstruction(
        name=name,
        smoothness_calculator=naive_piece_wise,
        reconstruction_error_measure=ReconstructionErrorMeasure(StencilCreatorFixedShape((3, 3)),
                                                                metric=metric,
                                                                central_cell_extra_weight=central_cell_extra_weight),
        refinement=refinement,
        cell_creators=
        [  # regular cell with piecewise_constant
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_condition_on_smoothness, value=REGULAR_CELL,
                                      condition=operator.eq),  # only regular cells
                orientator=BaseOrientator(dimensionality=2),
                stencil_creator=StencilCreatorFixedShape(stencil_shape=(1, 1)),
                cell_creator=PiecewiseConstantRegularCellCreator(
                    apriori_up_value=1, apriori_down_value=0, dimensionality=2)
            ),
            # curve cell
            CellCreatorPipeline(
                cell_iterator=partial(iterate_by_condition_on_smoothness, value=CURVE_CELL,
                                      condition=operator.eq),
                orientator=orientator,
                stencil_creator=stencil_creator,
                cell_creator=curve_cell_creator(regular_opposite_cell_searcher=get_opposite_regular_cells_by_minmax),
                reconstruction_error_measure=reconstruction_error_measure
            )
        ],
        obera_iterations=iterations
    )


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


def elvira():
    return get_sub_cell_model(ELVIRACurveCellCreator, 1, "ELVIRA", 0, 0, 2,
                              orientator=OrientByGradient(kernel_size=(5, 5), dimensionality=2, angle_threshold=0))


def elvira_w():
    return get_sub_cell_model(ELVIRACurveCellCreator, 1, "ELVIRA-W", 0, CCExtraWeight, 2,
                              orientator=OrientByGradient(kernel_size=(5, 5), dimensionality=2, angle_threshold=0))


def elvira_w_oriented():
    return get_sub_cell_model(ELVIRACurveCellCreator, 1, "ELVIRAGRAD", 0, CCExtraWeight, 2)


def elvira_w_oriented_l1():
    return get_sub_cell_model(ELVIRACurveCellCreator, 1, "ELVIRAGRAD", 0, CCExtraWeight, 1)


def elvira_w_oriented_ml():
    return get_sub_cell_model(ELVIRACurveCellCreator, 1, "ELVIRAGRAD", 0, CCExtraWeight, 2,
                              reconstruction_error_measure=ReconstructionErrorMeasureML(
                                  ml_model=kernel_lines_ml_model,
                                  stencil_creator=StencilCreatorFixedShape(
                                      kernel_lines_ml_model.dataset_manager.kernel_size),
                                  metric=2,
                                  central_cell_extra_weight=CCExtraWeight
                              ))


def elvira_go100_ref2():
    return get_sub_cell_model(ELVIRACurveCellCreator, 2, "ELVIRAGRAD", 0, CCExtraWeight, 2)


# It is key for OBERA Linear to use L1

def linear_obera():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=1, ccew=0),
                natural_params=True), 1,
        "LinearOpt", OBERA_ITERS, 0, 1)


# It is key for OBERA Linear to use L1

def linear_obera_w():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=1, ccew=0),
                natural_params=True), 1,
        "LinearOpt", OBERA_ITERS, CCExtraWeight, 1)


def linear_obera_w_l2():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=1, ccew=0),
                natural_params=True), 1,
        "LinearOpt", OBERA_ITERS, CCExtraWeight, 2)


def linear_obera_w_ml():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=1, ccew=0),
                natural_params=True), 1,
        "LinearOpt", OBERA_ITERS, CCExtraWeight, 1,
        reconstruction_error_measure=ReconstructionErrorMeasureML(
            ml_model=kernel_lines_ml_model,
            stencil_creator=StencilCreatorFixedShape(
                kernel_lines_ml_model.dataset_manager.kernel_size),
            metric=2,
            central_cell_extra_weight=CCExtraWeight
        )
    )


def linear_aero():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=0)), 1,
        "LinearAvg", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3))


def linear_aero_w():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=CCExtraWeight)), 1,
        "LinearAvg100", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3))


def linear_aero_consistent():
    return get_sub_cell_model(
        partial(ValuesLineConsistentCurveCellCreator, ccew=CCExtraWeight), 1,
        "LinearAvgConsistent", 0, CCExtraWeight, 1,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                               independent_dim_stencil_size=3))


def nn_linear():
    return get_sub_cell_model(partial(LearningCurveCellCreator, lines_ml_model), 1,
                              "LinearNN", 0, CCExtraWeight, 2)


def nn_quadratic_3x3():
    return get_sub_cell_model(partial(LearningCurveCellCreator, quadratics_ml_model), 1,
                              "QuadraticNN", 0, CCExtraWeight, 2,
                              )


def nn_quadratic_3x7():
    return get_sub_cell_model(partial(LearningCurveCellCreator, quadratics7_points_ml_model), 1,
                              "QuadraticNN", 0, CCExtraWeight, 2,
                              stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 7))
                              )


def nn_quadratic_3x7_params():
    return get_sub_cell_model(partial(LearningCurveCellCreator, quadratics7_params_ml_model), 1,
                              "QuadraticNN", 0, CCExtraWeight, 2,
                              stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 7))
                              )


def nn_quadratic_3x7_adapt():
    return get_sub_cell_model(partial(LearningCurveCellCreator, quadratics7_points_ml_model), 1,
                              "QuadraticNN", 0, CCExtraWeight, 2,
                              stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                                     independent_dim_stencil_size=3,
                                                                     dependent_dim_size=7)
                              )


def nn_quadratic_3x7_params_adapt():
    return get_sub_cell_model(partial(LearningCurveCellCreator, quadratics7_params_ml_model), 1,
                              "QuadraticNN", 0, CCExtraWeight, 2,
                              # stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 7))
                              stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                                     independent_dim_stencil_size=3,
                                                                     dependent_dim_size=7)
                              )


def quadratic_obera_non_adaptive():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticOptNonAdaptive", OBERA_ITERS, CCExtraWeight, 2)


def quadratic_obera_non_adaptive_l1():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticOptNonAdaptive", OBERA_ITERS, CCExtraWeight, 1)


def quadratic_obera():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticOpt", OBERA_ITERS, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3))


def quadratic_obera_l1():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticOpt", OBERA_ITERS, CCExtraWeight, 1,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3))


def quadratic_obera_ml_params():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=2, ccew=CCExtraWeight),
                natural_params=True), 1,
        "QuadraticOpt", OBERA_ITERS, CCExtraWeight, 2,
        stencil_creator=StencilCreatorFixedShape(kernel_quadratics_ml_model.dataset_manager.kernel_size),
        reconstruction_error_measure=ReconstructionErrorMeasureML(
            ml_model=[kernel_quadratics_ml_model],
            metric=2,
            central_cell_extra_weight=CCExtraWeight
        )
    )


def quadratic_obera_ml_points():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=2, ccew=CCExtraWeight),
                natural_params=False), 1,
        "QuadraticOpt", OBERA_ITERS, CCExtraWeight, 2,
        stencil_creator=StencilCreatorFixedShape(kernel_quadratics_points_ml_model.dataset_manager.kernel_size),
        reconstruction_error_measure=ReconstructionErrorMeasureML(
            ml_model=[kernel_quadratics_points_ml_model],
            metric=2,
            central_cell_extra_weight=CCExtraWeight
        )
    )


def quadratic_obera_ml_points_adapt():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=2, ccew=CCExtraWeight),
                natural_params=True), 1,
        "QuadraticOpt", OBERA_ITERS, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                               independent_dim_stencil_size=
                                               kernel_quadratics_points_ml_model.dataset_manager.kernel_size[0],
                                               dependent_dim_size=
                                               kernel_quadratics_points_ml_model.dataset_manager.kernel_size[1]),
        reconstruction_error_measure=ReconstructionErrorMeasureML(
            ml_model=kernel_quadratics_points_ml_model,
            stencil_creator=StencilCreatorAdaptive(
                smoothness_threshold=REGULAR_CELL,
                independent_dim_stencil_size=
                kernel_quadratics_points_ml_model.dataset_manager.kernel_size[0],
                dependent_dim_size=
                kernel_quadratics_points_ml_model.dataset_manager.kernel_size[1]),
            metric=2,
            central_cell_extra_weight=CCExtraWeight
        ))


def quadratic_aero():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticAvg", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3))


def cubic_aero():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=3, ccew=CCExtraWeight)), 1,
        "QuadraticAvg", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=5))


def cubic_obera():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=3, ccew=CCExtraWeight)), 1,
        "QuadraticOpt", OBERA_ITERS, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=5))


def quartic_aero():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=4, ccew=CCExtraWeight)), 1,
        "QuadraticAvg", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=5)
    )


def quartic_obera():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=4, ccew=CCExtraWeight)), 1,
        "QuadraticOpt", OBERA_ITERS, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=5)
    )


def quadratic_aero_ref2():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=2, ccew=CCExtraWeight)), 2,
        "QuadraticAvg", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3))


def obera_circle():
    return get_sub_cell_model(
        partial(TaylorCircleCurveCellCreator, ccew=CCExtraWeight, natural_params=False), 1,
        "CircleAvg", OBERA_ITERS, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3))


def obera_circle_vander():
    return get_sub_cell_model(ValuesCircleCellCreator, 1,
                              "CircleAvg", OBERA_ITERS, CCExtraWeight, 2,
                              stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                                     independent_dim_stencil_size=3))


def obera_circle_vander_ml():
    return get_sub_cell_model(ValuesCircleCellCreator, 1,
                              "CircleAvg", OBERA_ITERS, CCExtraWeight, 2,
                              stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                                     independent_dim_stencil_size=3),
                              # stencil_creator=StencilCreatorFixedShape(
                              #     kernel_circles_ml_model_points.dataset_manager.kernel_size),
                              reconstruction_error_measure=ReconstructionErrorMeasureML(
                                  ml_model=kernel_circles_ml_model_points,
                                  stencil_creator=StencilCreatorFixedShape(
                                      kernel_circles_ml_model_points.dataset_manager.kernel_size),
                                  metric=2, central_cell_extra_weight=CCExtraWeight
                              ))


if __name__ == "__main__":
    # ========== =========== ========== =========== #
    #                Experiment Run                 #
    # ========== =========== ========== =========== #
    data_manager = DataManager(
        path=config.results_path,
        name=f'AERO',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )
    # data_manager.load()

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "precompute_images",
        obtain_images
    )

    lab.define_new_block_of_functions(
        "precompute_error_resolution",
        obtain_image4error
    )

    lab.define_new_block_of_functions(
        "models",
        *list(map(fit_model,
                  [
                      piecewise_constant,
                      # elvira,
                      elvira_w_oriented,
                      elvira_w_oriented_l1,
                      # elvira_w_oriented_ml,

                      linear_obera,
                      linear_obera_w,
                      linear_obera_w_l2,
                      linear_obera_w_ml,

                      linear_aero,
                      linear_aero_w,
                      # linear_aero_consistent,

                      nn_linear,
                      # nn_quadratic_3x3,
                      # nn_quadratic_3x7,
                      # nn_quadratic_3x7_params,
                      nn_quadratic_3x7_adapt,
                      # nn_quadratic_3x7_params_adapt,

                      quadratic_obera_non_adaptive,
                      quadratic_obera_non_adaptive_l1,
                      quadratic_obera,
                      quadratic_obera_l1,
                      quadratic_obera_ml_params,
                      # quadratic_obera_ml_points_adapt,
                      quadratic_obera_ml_points,
                      quadratic_aero,

                      cubic_aero,
                      cubic_obera,

                      quartic_aero,
                      quartic_obera,

                      # elvira_go100_ref2,
                      # quadratic_aero_ref2,
                      #
                      # obera_circle,
                      obera_circle_vander,
                      # obera_circle_vander_ml
                  ]
                  )),
        recalculate=False
    )
    # num_cells_per_dim = np.logspace(np.log10(20), np.log10(100), num=20, dtype=int).tolist()[:1]
    num_cells_per_dim = np.logspace(np.log10(20), np.log10(100), num=20, dtype=int).tolist()
    num_cells_per_dim = np.logspace(np.log10(10), np.log10(20), num=5, dtype=int,
                                    endpoint=False).tolist() + num_cells_per_dim
    lab.execute(
        data_manager,
        num_cores=num_cores,
        forget=False,
        save_on_iteration=100,
        num_cells_per_dim=num_cells_per_dim,
        noise=[0],
        shape_name=[
            "Circle"
        ],
        sub_discretization2bound_error=[SUB_CELL_DISCRETIZATION2BOUND_ERROR],
    )


    # ========== =========== ========== =========== #
    #               Experiment Times                #
    # ========== =========== ========== =========== #
    def myround(n):
        # https://stackoverflow.com/questions/32812255/round-floats-down-in-python-to-keep-one-non-zero-decimal-only
        if n == 0:
            return 0
        sgn = -1 if n < 0 else 1
        scale = int(-np.floor(np.log10(abs(n))))
        if scale <= 0:
            scale = 1
        factor = 10 ** scale
        return sgn * np.floor(abs(n) * factor) / factor


    # times to fit cell
    df = next(make_data_frames(
        data_manager,
        var_names=["models", "time"],
        group_by=[],
        # models=models2plot,
        time=curve_cells_fitting_times,
    ))[1].groupby("models").apply(lambda x: np.nanmean(list(chain(*x["time"].values.tolist()))))
    runsinfo.append_info(
        **{k.replace("_", "-") + "-time": np.round(v, decimals=4) for k, v in df.items()}
    )

    # times to fit cell std
    dfstd = next(make_data_frames(
        data_manager,
        var_names=["models", "time"],
        group_by=[],
        # models=models2plot,
        time=curve_cells_fitting_times,
    ))[1].groupby("models").apply(lambda x: np.nanstd(list(chain(*x["time"].values.tolist()))))
    runsinfo.append_info(
        **{"std-" + k.replace("_", "-") + "-time": np.round(v, decimals=4) for k, v in dfstd.items()}
    )

    dfstd = next(make_data_frames(
        data_manager,
        var_names=["models", "time"],
        group_by=[],
        # models=models2plot,
        time=curve_cells_fitting_times,
    ))[1].groupby("models").apply(lambda x: np.nanquantile(list(chain(*x["time"].values.tolist())), 0.05))
    runsinfo.append_info(
        **{"qlow-" + k.replace("_", "-") + "-time": np.round(v, decimals=4) for k, v in dfstd.items()}
    )

    dfstd = next(make_data_frames(
        data_manager,
        var_names=["models", "time"],
        group_by=[],
        # models=models2plot,
        time=curve_cells_fitting_times,
    ))[1].groupby("models").apply(lambda x: np.nanquantile(list(chain(*x["time"].values.tolist())), 0.95))
    runsinfo.append_info(
        **{"qhigh-" + k.replace("_", "-") + "-time": np.round(v, decimals=4) for k, v in dfstd.items()}
    )

    dfstd = next(make_data_frames(
        data_manager,
        var_names=["models", "time"],
        group_by=[],
        # models=models2plot,
        time=curve_cells_fitting_times,
    ))[1].groupby("models").apply(lambda x: np.nanquantile(list(chain(*x["time"].values.tolist())), 0.5))
    runsinfo.append_info(
        **{"median-" + k.replace("_", "-") + "-time": np.round(v, decimals=4) for k, v in dfstd.items()}
    )

    # ========== =========== ========== =========== #
    #               Experiment Plots                #
    # ========== =========== ========== =========== #
    for group, model_style in accepted_models.items():
        models2plot = list(model_style.keys())
        palette = {names_dict[k]: v.color for k, v in model_style.items()}

        for er in ["l1", "linf"]:
            plot_h_convergence(
                data_manager,
                path=config.subcell_paper_figures_path,
                name=f"HConvergence_{er}_{group}",
                folder=group,
                log="xy",
                error=er,
                models=models2plot,
                model_style=model_style,
                names_dict=names_dict,
                sorted_models=lambda models: models2plot.index(models),
                sort_by=["sorted_models"],
                format=".pdf",
                rateonly=rateonly,
            )

        generic_plot(data_manager,
                     name=f"Convergence_{group}",
                     path=config.subcell_paper_figures_path,
                     folder=group,
                     x="N", y="error_l1", label="models", num_cells_per_dim=num_cells_per_dim,
                     plot_func=NamedPartial(plot_convergence, model_style=model_style, names_dict=names_dict),
                     log="xy",
                     N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     models=models2plot,
                     method=lambda models: names_dict[str(models)],
                     names_dict=names_dict,
                     sorted_models=lambda models: models2plot.index(models),
                     sort_by=['sorted_models'],
                     format=".pdf",
                     )

        generic_plot(data_manager,
                     name=f"TimeComplexityPerCellBar_{group}",
                     path=config.subcell_paper_figures_path,
                     folder=group,
                     x="method", y="time", num_cells_per_dim=num_cells_per_dim,
                     plot_func=NamedPartial(sns.boxenplot, palette=palette),
                     log="y",
                     time=curve_cells_fitting_times,
                     N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     models=models2plot,
                     method=lambda models: names_dict[str(models)],
                     sorted_models=lambda models: models2plot.index(models),
                     sort_by=["sorted_models"],
                     format=".pdf"
                     )

        generic_plot(data_manager,
                     name=f"TimeComplexity_{group}",
                     path=config.subcell_paper_figures_path,
                     folder=group,
                     x="N", y="time", label="method", num_cells_per_dim=num_cells_per_dim,
                     plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--", palette=palette),
                     log="xy",
                     time=lambda time_to_fit: time_to_fit,
                     N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     models=models2plot,
                     method=lambda models: names_dict[str(models)],
                     sorted_models=lambda models: models2plot.index(models),
                     sort_by=['sorted_models'],
                     format=".pdf"
                     )
        generic_plot(data_manager,
                     name=f"TimeComplexityPerCell_{group}",
                     path=config.subcell_paper_figures_path,
                     folder=group,
                     x="N", y="time", label="method", num_cells_per_dim=num_cells_per_dim,
                     plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--", palette=palette),
                     log="xy",
                     time=curve_cells_fitting_times,
                     N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     models=models2plot,
                     method=lambda models: names_dict[str(models)],
                     sorted_models=lambda models: models2plot.index(models),
                     sort_by=['sorted_models'],
                     format=".pdf"
                     )

        generic_plot(data_manager,
                     name=f"TimeComplexityVSError_{group}",
                     path=config.subcell_paper_figures_path,
                     folder=group,
                     x="time", y="error_l1", label="method", num_cells_per_dim=num_cells_per_dim,
                     plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--", palette=palette),
                     log="xy", time=lambda time_to_fit: time_to_fit, N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     models=models2plot,
                     method=lambda models: names_dict[str(models)],
                     sorted_models=lambda models: models2plot.index(models),
                     sort_by=['sorted_models'],
                     format=".pdf"
                     )

    num_cells_per_dim_2plot = [num_cells_per_dim[0], num_cells_per_dim[5], num_cells_per_dim[10]]
    for group, model_style in accepted_models.items():
        models2plot = list(model_style.keys())
        plot_reconstruction(
            data_manager,
            folder=group,
            name=f"{group}",
            axes_by=['models'],
            models=models2plot,
            plot_by=['num_cells_per_dim'],
            num_cells_per_dim=num_cells_per_dim_2plot,
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

    print("CO2 consumption: ", data_manager.CO2kg)
    copy_main_script_version(__file__, data_manager.path)
