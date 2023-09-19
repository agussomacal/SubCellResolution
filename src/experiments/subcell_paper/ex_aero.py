import time
from functools import partial

import numpy as np
import seaborn as sns
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline, FunctionBlock
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot
from experiments.LearningMethods import flatter, sktorch_20x20_relu, skkeras_20x20_relu
from experiments.subcell_paper.global_params import SUB_CELL_DISCRETIZATION2BOUND_ERROR, OBERA_ITERS, \
    CCExtraWeight, runsinfo, cblue, corange, cgreen, cred, cpurple, cbrown, cpink, cgray, cyellow, ccyan
from experiments.subcell_paper.obera_experiments import get_sub_cell_model, get_shape, plot_reconstruction
from experiments.subcell_paper.tools import calculate_averages_from_image, calculate_averages_from_curve
from lib.AuxiliaryStructures.Constants import REGULAR_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.LearningCurveCellCreator import LearningCurveCellCreator
from lib.CellCreators.CurveCellCreators.TaylorCurveCellCreator import TaylorCircleCurveCellCreator
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator, \
    ValuesLineConsistentCurveCellCreator, ValuesCircleCellCreator
from lib.CellCreators.RegularCellCreator import MirrorCellCreator
from lib.CellIterators import iterate_all
from lib.CellOrientators import BaseOrientator, OrientPredefined, OrientByGradient
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.Curves.VanderCurves import CurveVandermondePolynomial
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import CURVE_PROBLEM
from lib.DataManagers.DatasetsManagers.DatasetsManagerLinearCurves import DatasetsManagerLinearCurves, ANGLE_OBJECTIVE
from lib.DataManagers.LearningMethodManager import LearningMethodManager
from lib.SmoothnessCalculators import indifferent
from lib.StencilCreators import StencilCreatorFixedShape, StencilCreatorAdaptive
from lib.SubCellReconstruction import SubCellReconstruction, ReconstructionErrorMeasureBase, CellCreatorPipeline

N = int(1e6)
workers = 10
dataset_manager_3_8pi = DatasetsManagerLinearCurves(
    velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=workers, recalculate=False, learning_objective=ANGLE_OBJECTIVE, angle_limits=(-3 / 8, 3 / 8),
    value_up_random=False
)

nnlm = LearningMethodManager(
    dataset_manager=dataset_manager_3_8pi,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=Pipeline(
        [
            ("Flatter", FunctionTransformer(flatter)),
            ("NN", MLPRegressor(hidden_layer_sizes=(20, 20,), activation='relu', learning_rate_init=0.1,
                                learning_rate="adaptive", solver="lbfgs"))
        ]
    ),
    refit=False, n2use=-1,
    training_noise=1e-5, train_percentage=0.9
)

nnlmkeras = LearningMethodManager(
    dataset_manager=dataset_manager_3_8pi,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=skkeras_20x20_relu,
    refit=False, n2use=-1,
    training_noise=1e-5, train_percentage=0.9
)


# nnlmtorch = LearningMethodManager(
#     dataset_manager=dataset_manager_3_8pi,
#     type_of_problem=CURVE_PROBLEM,
#     trainable_model=sktorch_20x20_relu,
#     refit=False, n2use=-1,
#     training_noise=1e-5, train_percentage=0.9
# )


def fit_model(sub_cell_model):
    def decorated_func(image, noise, sub_discretization2bound_error):
        # image = load_image(image)
        # avg_values = calculate_averages_from_image(image, num_cells_per_dim)
        np.random.seed(42)
        avg_values = image + np.random.uniform(-noise, noise, size=image.shape)

        model = sub_cell_model()
        print(f"Doing model: {decorated_func.__name__}")

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
def elvira():
    return get_sub_cell_model(ELVIRACurveCellCreator, 1, "ELVIRA", 0, 0, 2,
                              orientator=OrientByGradient(kernel_size=(5, 5), dimensionality=2, angle_threshold=0))


@fit_model
def elvira_w():
    return get_sub_cell_model(ELVIRACurveCellCreator, 1, "ELVIRA-W", 0, CCExtraWeight, 2,
                              orientator=OrientByGradient(kernel_size=(5, 5), dimensionality=2, angle_threshold=0))


@fit_model
def elvira_w_oriented():
    return get_sub_cell_model(ELVIRACurveCellCreator, 1, "ELVIRAGRAD", 0, CCExtraWeight, 2)


@fit_model
def elvira_go100_ref2():
    return get_sub_cell_model(ELVIRACurveCellCreator, 2, "ELVIRAGRAD", 0, CCExtraWeight, 2)


# It is key for OBERA Linear to use L1
@fit_model
def linear_obera():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=1, ccew=0),
                natural_params=True), 1,
        "LinearOpt", OBERA_ITERS, 0, 1)


# It is key for OBERA Linear to use L1
@fit_model
def linear_obera_w():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=1, ccew=0),
                natural_params=True), 1,
        "LinearOpt", OBERA_ITERS, CCExtraWeight, 1)


@fit_model
def linear_aero():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=0)), 1,
        "LinearAvg", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3,
                                               center_weight=2.1))


@fit_model
def linear_aero_w():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=CCExtraWeight)), 1,
        "LinearAvg100", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3,
                                               center_weight=2.1))


@fit_model
def linear_aero_consistent():
    return get_sub_cell_model(partial(ValuesLineConsistentCurveCellCreator, ccew=CCExtraWeight), 1,
                              "LinearAvgConsistent", 0, CCExtraWeight, 1,
                              stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                                     independent_dim_stencil_size=3,
                                                                     center_weight=2.1))


@fit_model
def nn_linear():
    return get_sub_cell_model(partial(LearningCurveCellCreator, nnlm), 1,
                              "LinearNN", 0, CCExtraWeight, 2)


# @fit_model
# def nn_linear_torch():
#     return get_sub_cell_model(partial(LearningCurveCellCreator, nnlmtorch), 1,
#                               "LinearNN", 0, CCExtraWeight, 2)


# @fit_model
# def nn_linear_keras():
#     return get_sub_cell_model(partial(LearningCurveCellCreator, nnlmkeras), 1,
#                               "LinearNN", 0, CCExtraWeight, 2)


@fit_model
def quadratic_obera_non_adaptive():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticOptNonAdaptive", OBERA_ITERS, CCExtraWeight, 2)


@fit_model
def quadratic_obera():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticOpt", OBERA_ITERS, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3))


@fit_model
def quadratic_aero():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticAvg", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3))


@fit_model
def quadratic_aero_ref2():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=2, ccew=CCExtraWeight)), 2,
        "QuadraticAvg", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3))


@fit_model
def obera_circle():
    return get_sub_cell_model(
        partial(TaylorCircleCurveCellCreator, ccew=CCExtraWeight, natural_params=False), 1,
        "CircleAvg", OBERA_ITERS, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3))


@fit_model
def obera_circle_vander():
    return get_sub_cell_model(ValuesCircleCellCreator, 1,
                              "CircleAvg", OBERA_ITERS, CCExtraWeight, 2,
                              stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0,
                                                                     independent_dim_stencil_size=3))


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name=f'AERO',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )
    data_manager.load()

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
        piecewise_constant,
        elvira,
        elvira_w_oriented,
        linear_obera,
        linear_obera_w,
        linear_aero,
        linear_aero_w,
        linear_aero_consistent,
        nn_linear,
        # nn_linear_torch,
        # nn_linear_keras,

        quadratic_obera_non_adaptive,
        quadratic_obera,
        quadratic_aero,

        # elvira_go100_ref2,
        # quadratic_aero_ref2,

        # obera_circle,
        # obera_circle_vander,
        recalculate=True
    )
    # num_cells_per_dim = np.logspace(np.log10(10), np.log10(100), num=20, dtype=int).tolist()[:5]
    num_cells_per_dim = np.logspace(np.log10(20), np.log10(100), num=20, dtype=int).tolist()
    num_cells_per_dim = np.logspace(np.log10(10), np.log10(20), num=5, dtype=int,
                                    endpoint=False).tolist() + num_cells_per_dim
    num_cores = 15
    lab.execute(
        data_manager,
        num_cores=num_cores,
        forget=False,
        save_on_iteration=None,
        num_cells_per_dim=num_cells_per_dim,
        noise=[0],
        shape_name=[
            "Circle"
        ],
        sub_discretization2bound_error=[SUB_CELL_DISCRETIZATION2BOUND_ERROR],
    )

    names_dict = {
        "piecewise_constant": "Piecewise Constant",
        "nn_linear": "NN Linear",
        "nn_linear_torch": "NN Linear Torch",
        "nn_linear_keras": "NN Linear Keras",
        "elvira": "ELVIRA",
        "elvira_w": "ELVIRA-W",
        "elvira_w_oriented": "ELVIRA-W Oriented",
        "linear_obera": "OBERA Linear",
        "linear_obera_w": "OBERA-W Linear",
        "linear_aero": "AERO Linear",
        "linear_aero_w": "AERO-W Linear",
        "linear_aero_consistent": "AERO Linear Column Consistent",
        "quadratic_obera_non_adaptive": "OBERA Quadratic 3x3",
        "quadratic_obera": "OBERA Quadratic",
        "quadratic_aero": "AERO Quadratic",
        "obera_circle": "OBERA Circle",
        "obera_circle_vander": "OBERA Circle ReParam",
        # "elvira_go100_ref2",
        # "quadratic_aero_ref2",
        # "circle_avg",
        # "circle_vander_avg",
    }
    model_color = {
        "piecewise_constant": cpink,
        "nn_linear": cgreen,
        "nn_linear_torch": "forestgreen",
        "nn_linear_keras": "mediumseagreen",
        "elvira": cyellow,
        "elvira_w": None,
        "elvira_w_oriented": corange,
        "linear_obera": cbrown,
        "linear_obera_w": cpurple,
        "linear_aero": ccyan,
        "linear_aero_w": cblue,
        "linear_aero_consistent": cpink,

        "quadratic_obera_non_adaptive": cgray,
        "quadratic_obera": cred,
        "quadratic_aero": cgreen,
        "obera_circle": cyellow,
        "obera_circle_vander": cpurple,
    }

    runsinfo.append_info(
        **{k.replace("_", "-"): v for k, v in names_dict.items()}
    )

    # -------------------- linear models -------------------- #
    accepted_models = {
        "LinearModels": [
            "nn_linear",
            "nn_linear_torch",
            "nn_linear_keras",
            "elvira",
            "elvira_w_oriented",
            "linear_obera",
            "linear_obera_w",
            "linear_aero",
            "linear_aero_w",
            "linear_aero_consistent"
        ],
        "HighOrderModels": [
            "piecewise_constant",
            "linear_aero_w",
            "quadratic_obera_non_adaptive",
            "quadratic_obera",
            "quadratic_aero",
            "obera_circle",
            "obera_circle_vander"
        ]
    }
    rateonly = ["piecewise_constant", "quadratic_aero", "linear_aero_w", "elvira", "elvira_w_oriented", "linear_obera"]

    error = lambda reconstruction, image4error: (
        np.sqrt(np.mean(np.abs(reconstruction - image4error).ravel() ** 2))) if reconstruction is not None else np.nan


    def plot_convergence(data, x, y, hue, ax, threshold=25, rateonly=rateonly, *args, **kwargs):
        # sns.scatterplot(data=data, x=x, y=y, hue=label, ax=ax)
        for method, df in data.groupby(hue):
            name = f"{names_dict[str(method)]}"
            if method in rateonly:
                hinv = np.sqrt(df[x].values)
                valid_ix = hinv > threshold
                rate, origin = np.ravel(np.linalg.lstsq(
                    np.vstack([np.log(hinv[valid_ix]), np.ones(np.sum(valid_ix))]).T,
                    np.log(df[y].values[valid_ix]).reshape((-1, 1)), rcond=None)[0])
                # ax.plot(df[x].values[valid_ix], np.sqrt(df[x].values[valid_ix]) ** rate * np.exp(origin), "-",
                #         c=model_color[method], linewidth=3, alpha=0.5)
                name = name + f": O({abs(rate):.1f})"
            sns.lineplot(x=df[x], y=df[y], color=model_color[method], label=name, ax=ax,
                         marker="o", linestyle="--", alpha=1)
            # ax.plot(df[x], df[y], marker=".", linestyle=":", c=model_color[method], label=name)
        ax.set_xlabel(fr"${{{x}}}$")
        ax.set_ylabel(r"$||u-\tilde u ||_{L_2}$")

        # n = np.sort(np.unique(data[x]))
        # ax.plot(n, 1 / n, ":", c=cgray, linewidth=2, alpha=0.5, label=r"$O(h^{-1})$")
        # ax.plot(n, 1 / n ** 2, ":", c=cgray, linewidth=2, alpha=0.5, label=r"$O(h^{-2})$")
        # ax.plot(n, 1 / n ** 3, ":", c=cgray, linewidth=2, alpha=0.5, label=r"$O(h^{-3})$")


    for group, models2plot in accepted_models.items():
        generic_plot(data_manager,
                     name=f"Convergence_{group}",
                     path=config.subcell_paper_figures_path,
                     folder=group,
                     x="N", y="error", label="models", num_cells_per_dim=num_cells_per_dim,
                     plot_func=plot_convergence,
                     # plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                     log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     error=error,
                     models=models2plot,
                     sort_by=["models", "N"],
                     method=lambda models: names_dict[str(models)]
                     )

        generic_plot(data_manager,
                     name=f"HConvergence_{group}",
                     folder=group,
                     x="h", y="error", label="models", num_cells_per_dim=num_cells_per_dim,
                     plot_func=NamedPartial(plot_convergence, threshold=1 / 25,
                                            palette={names_dict[k]: v for k, v in model_color.items()}),
                     # plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                     log="xy", h=lambda num_cells_per_dim: 1 / num_cells_per_dim,
                     error=error,
                     models=models2plot,
                     sort_by=["models", "h"],
                     method=lambda models: names_dict[str(models)]
                     )

        generic_plot(data_manager,
                     name=f"TimeComplexity_{group}",
                     path=config.subcell_paper_figures_path,
                     folder=group,
                     x="N", y="time", label="method", num_cells_per_dim=num_cells_per_dim,
                     plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--",
                                            palette={names_dict[k]: v for k, v in model_color.items()}),
                     log="xy", time=lambda time_to_fit: time_to_fit, N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     error=error,
                     models=models2plot,
                     method=lambda models: names_dict[str(models)]
                     )

        generic_plot(data_manager,
                     name=f"TimeComplexityMSE_{group}",
                     path=config.subcell_paper_figures_path,
                     folder=group,
                     x="time", y="error", label="method", num_cells_per_dim=num_cells_per_dim,
                     plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--",
                                            palette={names_dict[k]: v for k, v in model_color.items()}),
                     log="xy", time=lambda time_to_fit: time_to_fit, N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     error=error,
                     models=models2plot,
                     method=lambda models: names_dict[str(models)]
                     )

        plot_reconstruction(
            data_manager,
            folder=group,
            name=f"{group}",
            axes_by=['models'],
            models=models2plot,
            plot_by=['num_cells_per_dim'],
            axes_xy_proportions=(15, 15),
            difference=False,
            plot_curve=True,
            plot_curve_winner=False,
            plot_vh_classification=False,
            plot_singular_cells=False,
            plot_original_image=True,
            numbers_on=True,
            plot_again=False,
            num_cores=1,
            # trim=trim
        )

    print("CO2 consumption: ", data_manager.CO2kg)
    copy_main_script_version(__file__, data_manager.path)
