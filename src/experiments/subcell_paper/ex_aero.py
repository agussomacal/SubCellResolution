import time
from functools import partial
from itertools import chain

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot, make_data_frames, perplex_plot
from experiments.MLTraining.ml_cell_averages import kernel_circles_ml_model_points, kernel_quadratics_avg_ml_model, \
    kernel_lines_ml_model, kernel_quadratics_ml_model
from experiments.MLTraining.ml_curve_params import lines_ml_model, quadratics7_points_ml_model, \
    quadratics7_params_ml_model, quadratics_ml_model
from experiments.subcell_paper.global_params import SUB_CELL_DISCRETIZATION2BOUND_ERROR, OBERA_ITERS, \
    CCExtraWeight, runsinfo, cblue, corange, cgreen, cred, cpurple, cbrown, cpink, cgray, cyellow, ccyan
from experiments.subcell_paper.obera_experiments import get_sub_cell_model, get_shape, plot_reconstruction
from experiments.subcell_paper.tools import calculate_averages_from_curve, \
    curve_cells_fitting_times, singular_cells_mask, make_image_high_resolution
from lib.AuxiliaryStructures.Constants import REGULAR_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.LearningCurveCellCreator import LearningCurveCellCreator
from lib.CellCreators.CurveCellCreators.TaylorCurveCellCreator import TaylorCircleCurveCellCreator
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator, \
    ValuesLineConsistentCurveCellCreator, ValuesCircleCellCreator
from lib.CellCreators.RegularCellCreator import MirrorCellCreator
from lib.CellIterators import iterate_all
from lib.CellOrientators import BaseOrientator, OrientByGradient
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.Curves.VanderCurves import CurveVandermondePolynomial
from lib.SmoothnessCalculators import indifferent
from lib.StencilCreators import StencilCreatorFixedShape, StencilCreatorAdaptive
from lib.SubCellReconstruction import SubCellReconstruction, ReconstructionErrorMeasureBase, CellCreatorPipeline, \
    ReconstructionErrorMeasureML, reconstruct_by_factor


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


def fit_model(sub_cell_model):
    def decorated_func(image, noise, sub_discretization2bound_error):
        np.random.seed(42)
        avg_values = image + np.random.uniform(-noise, noise, size=image.shape)

        model = sub_cell_model()
        print(f"Doing model: {decorated_func.__name__}")

        t0 = time.time()
        model.fit(average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"))
        t_fit = time.time() - t0

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


def linear_aero():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=0)), 1,
        "LinearAvg", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3,
                                               center_weight=2.1))


def linear_aero_w():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=CCExtraWeight)), 1,
        "LinearAvg100", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3,
                                               center_weight=2.1))


def linear_aero_consistent():
    return get_sub_cell_model(partial(ValuesLineConsistentCurveCellCreator, ccew=CCExtraWeight), 1,
                              "LinearAvgConsistent", 0, CCExtraWeight, 1,
                              stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL,
                                                                     independent_dim_stencil_size=3,
                                                                     center_weight=2.1))


def nn_linear():
    return get_sub_cell_model(partial(LearningCurveCellCreator, lines_ml_model), 1,
                              "LinearNN", 0, CCExtraWeight, 2)


def nn_quadratic_3x3():
    return get_sub_cell_model(partial(LearningCurveCellCreator, quadratics_ml_model), 1,
                              "QuadraticNN", 0, CCExtraWeight, 2,
                              # stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0,
                              #                                        independent_dim_stencil_size=3,
                              #                                        dependent_dim_size=7),
                              )


def nn_quadratic_3x7():
    return get_sub_cell_model(partial(LearningCurveCellCreator, quadratics7_points_ml_model), 1,
                              "QuadraticNN", 0, CCExtraWeight, 2,
                              stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 7))
                              # stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0,
                              #                                        independent_dim_stencil_size=3,
                              #                                        dependent_dim_size=7)
                              )


def nn_quadratic_3x7_params():
    return get_sub_cell_model(partial(LearningCurveCellCreator, quadratics7_params_ml_model), 1,
                              "QuadraticNN", 0, CCExtraWeight, 2,
                              stencil_creator=StencilCreatorFixedShape(stencil_shape=(3, 7))
                              # stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0,
                              #                                        independent_dim_stencil_size=3,
                              #                                        dependent_dim_size=7)
                              )


def quadratic_obera_non_adaptive():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticOptNonAdaptive", OBERA_ITERS, CCExtraWeight, 2)


def quadratic_obera():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticOpt", OBERA_ITERS, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3))


def quadratic_obera_ml():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveVandermondePolynomial, degree=2, ccew=CCExtraWeight),
                natural_params=True), 1,
        "QuadraticOpt", OBERA_ITERS, CCExtraWeight, 2,
        # stencil_creator=StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3),
        stencil_creator=StencilCreatorFixedShape(kernel_quadratics_avg_ml_model.dataset_manager.kernel_size),
        reconstruction_error_measure=ReconstructionErrorMeasureML(
            ml_model=kernel_quadratics_ml_model,
            stencil_creator=StencilCreatorFixedShape(
                kernel_quadratics_ml_model.dataset_manager.kernel_size),
            metric=2,
            central_cell_extra_weight=CCExtraWeight
        ))


def quadratic_aero():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=2, ccew=CCExtraWeight)), 1,
        "QuadraticAvg", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3))


def quadratic_aero_ref2():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator,
                vander_curve=partial(CurveAveragePolynomial, degree=2, ccew=CCExtraWeight)), 2,
        "QuadraticAvg", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3))


def obera_circle():
    return get_sub_cell_model(
        partial(TaylorCircleCurveCellCreator, ccew=CCExtraWeight, natural_params=False), 1,
        "CircleAvg", OBERA_ITERS, CCExtraWeight, 2,
        stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0, independent_dim_stencil_size=3))


def obera_circle_vander():
    return get_sub_cell_model(ValuesCircleCellCreator, 1,
                              "CircleAvg", OBERA_ITERS, CCExtraWeight, 2,
                              stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0,
                                                                     independent_dim_stencil_size=3))


def obera_circle_vander_ml():
    return get_sub_cell_model(ValuesCircleCellCreator, 1,
                              "CircleAvg", OBERA_ITERS, CCExtraWeight, 2,
                              # stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0,
                              #                                        independent_dim_stencil_size=3),
                              stencil_creator=StencilCreatorFixedShape(
                                      kernel_circles_ml_model_points.dataset_manager.kernel_size),
                              reconstruction_error_measure=ReconstructionErrorMeasureML(
                                  ml_model=kernel_circles_ml_model_points,
                                  stencil_creator=StencilCreatorFixedShape(
                                      kernel_circles_ml_model_points.dataset_manager.kernel_size),
                                  metric=2,
                                  central_cell_extra_weight=CCExtraWeight
                              ))


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        # name=f'AERO',
        name=f'AERO_NN2',
        format=JOBLIB,
        trackCO2=True,
        country_alpha_code="FR"
    )
    data_manager.load()

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
                      # piecewise_constant,
                      # elvira,
                      # elvira_w_oriented,
                      # elvira_w_oriented_ml,
                      #
                      # linear_obera,
                      # linear_obera_w,
                      # linear_aero,
                      # linear_aero_w,
                      # linear_aero_consistent,

                      # nn_linear,
                      # nn_quadratic_3x3,
                      # nn_quadratic_3x7,
                      # nn_quadratic_3x7_params,

                      # quadratic_obera_non_adaptive,
                      # quadratic_obera,
                      quadratic_obera_ml,
                      # quadratic_aero,

                      # elvira_go100_ref2,
                      # quadratic_aero_ref2,

                      # obera_circle,
                      # obera_circle_vander,
                      # obera_circle_vander_ml
                  ]
                  )),
        recalculate=True
    )
    num_cells_per_dim = np.logspace(np.log10(10), np.log10(100), num=20, dtype=int).tolist()[:5]
    # num_cells_per_dim = np.logspace(np.log10(20), np.log10(100), num=20, dtype=int).tolist()
    # num_cells_per_dim = np.logspace(np.log10(20), np.log10(500), num=30, dtype=int).tolist()
    # num_cells_per_dim = np.logspace(np.log10(10), np.log10(20), num=5, dtype=int,
    #                                 endpoint=False).tolist() + num_cells_per_dim
    num_cores = 1
    lab.execute(
        data_manager,
        num_cores=num_cores,
        forget=False,
        save_on_iteration=5,
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
        "nn_quadratic_3x3": "NN Quadratic 3x3",
        "nn_quadratic_3x7": "NN Quadratic 3x7",
        "nn_quadratic_3x7_params": "NN Quadratic 3x7 params",

        "elvira": "ELVIRA",
        "elvira_w": "ELVIRA-W",
        "elvira_w_oriented": "ELVIRA-W Oriented",
        "elvira_w_oriented_ml": "ELVIRA-W Oriented ML",

        "linear_obera": "OBERA Linear",
        "linear_obera_w": "OBERA-W Linear",
        "linear_aero": "AEROS Linear",
        "linear_aero_w": "AEROS-W Linear",
        "linear_aero_consistent": "AEROS Linear Column Consistent",

        "quadratic_obera_non_adaptive": "OBERA Quadratic 3x3",
        "quadratic_obera": "OBERA Quadratic",
        "quadratic_obera_ml": "OBERA Quadratic ML",
        "quadratic_aero": "AEROS Quadratic",

        "obera_circle": "OBERA Circle",
        "obera_circle_vander": "OBERA Circle ReParam",
        "obera_circle_vander_ml": "OBERA Circle ReParam ML",
        # "elvira_go100_ref2",
        # "quadratic_aero_ref2",
    }
    model_color = {
        "piecewise_constant": cpink,
        "nn_linear": "forestgreen",
        "nn_quadratic_3x3": cpurple,
        "nn_quadratic_3x7": cblue,
        "nn_quadratic_3x7_params": ccyan,
        "elvira": cyellow,
        "elvira_w": None,
        "elvira_w_oriented": corange,
        "elvira_w_oriented_ml": cred,
        "linear_obera": cbrown,
        "linear_obera_w": cpurple,
        "linear_aero": ccyan,
        "linear_aero_w": cblue,
        "linear_aero_consistent": cpink,

        "quadratic_obera_non_adaptive": cgray,
        "quadratic_obera": cred,
        "quadratic_obera_ml": corange,
        "quadratic_aero": cgreen,
        "obera_circle": cyellow,
        "obera_circle_vander": cpurple,
        "obera_circle_vander_ml": "mediumseagreen",
    }

    runsinfo.append_info(
        **{k.replace("_", "-"): v for k, v in names_dict.items()}
    )

    # -------------------- linear models -------------------- #
    accepted_models = {
        # "LinearModels": [
        #     "nn_linear",
        #     "elvira",
        #     "elvira_w_oriented",
        #     "elvira_w_oriented_ml"
        #     "linear_obera",
        #     "linear_obera_w",
        #     "linear_aero",
        #     "linear_aero_w",
        #     "linear_aero_consistent"
        # ],
        # "HighOrderModels": [
        #     "piecewise_constant",
        #     "linear_aero_w",
        #     "quadratic_obera_non_adaptive",
        #     "quadratic_obera",
        #     "quadratic_aero",
        #     "obera_circle",
        #     "obera_circle_vander",
        # ],
        "OtherTests": [
            "piecewise_constant",
            "nn_linear",
            "linear_aero_w",
            "quadratic_obera_non_adaptive",
            "quadratic_obera",
            "quadratic_obera_ml",
            "quadratic_aero",
            "obera_circle",
            "obera_circle_vander",
            "obera_circle_vander_ml",
            "nn_quadratic_3x3",
            "nn_quadratic_3x7",
            "nn_quadratic_3x7_params",
        ]
    }
    rateonly = list(names_dict.keys())[:-2]

    error = lambda reconstruction, image4error: np.mean(np.abs(reconstruction - image4error).ravel()) \
        if reconstruction is not None else np.nan


    @perplex_plot(group_by="models")
    def plot_h_convergence(fig, ax, num_cells_per_dim, reconstruction, image4error, models,
                           threshold=1.0 / np.sqrt(1000), rateonly=None, *args, **kwargs):
        er = np.array(list(map(lambda x: error(*x), zip(reconstruction, image4error))))
        name = f"{names_dict[str(models)]}"
        h = 1.0 / np.array(num_cells_per_dim)
        if rateonly is None or models in rateonly:
            valid_ix = h < threshold
            rate, origin = np.ravel(np.linalg.lstsq(
                np.vstack([np.log(h[valid_ix]), np.ones(np.sum(valid_ix))]).T,
                np.log(er[valid_ix]).reshape((-1, 1)), rcond=None)[0])
            # ax.plot(df[x].values[valid_ix], np.sqrt(df[x].values[valid_ix]) ** rate * np.exp(origin), "-",
            #         c=model_color[method], linewidth=3, alpha=0.5)
            name = name + f": O({abs(rate):.1f})"
        sns.lineplot(x=h, y=er, color=model_color[models], label=name, ax=ax,
                     marker="o", linestyle="--", alpha=1)
        # ax.plot(df[x], df[y], marker=".", linestyle=":", c=model_color[method], label=name)
        ax.set_xlabel(fr"$h$")
        ax.set_ylabel(r"$||u-\tilde u ||_{L^1}$")
        ax.set_xscale("log")
        ax.set_yscale("log")


    def plot_convergence(data, x, y, hue, ax, threshold=np.sqrt(1000), rateonly=None, *args, **kwargs):
        # sns.scatterplot(data=data, x=x, y=y, hue=label, ax=ax)
        for method, df in data.groupby(hue):
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
            sns.lineplot(x=df[x], y=df[y], color=model_color[method], label=name, ax=ax,
                         marker="o", linestyle="--", alpha=1)
            # ax.plot(df[x], df[y], marker=".", linestyle=":", c=model_color[method], label=name)
        ax.set_xlabel(fr"${{{x}}}$")
        ax.set_ylabel(r"$||u-\tilde u ||_{L^1}$")

        # n = np.sort(np.unique(data[x]))
        # ax.plot(n, 1 / n, ":", c=cgray, linewidth=2, alpha=0.5, label=r"$O(h^{-1})$")
        # ax.plot(n, 1 / n ** 2, ":", c=cgray, linewidth=2, alpha=0.5, label=r"$O(h^{-2})$")
        # ax.plot(n, 1 / n ** 3, ":", c=cgray, linewidth=2, alpha=0.5, label=r"$O(h^{-3})$")


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

    for group, models2plot in accepted_models.items():
        generic_plot(data_manager,
                     name=f"TimeComplexityPerCellBar_{group}",
                     path=config.subcell_paper_figures_path,
                     folder=group,
                     x="method", y="time", num_cells_per_dim=num_cells_per_dim,
                     plot_func=NamedPartial(sns.boxenplot,
                                            palette={names_dict[k]: v for k, v in model_color.items()}),
                     log="y",
                     time=curve_cells_fitting_times,
                     N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     error=error,
                     models=models2plot,
                     method=lambda models: names_dict[str(models)],
                     format=".pdf"
                     )

        generic_plot(data_manager,
                     name=f"Convergence_{group}",
                     path=config.subcell_paper_figures_path,
                     folder=group,
                     x="N", y="error", label="models", num_cells_per_dim=num_cells_per_dim,
                     plot_func=plot_convergence,
                     # plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                     log="xy",
                     N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     error=error,
                     models=models2plot,
                     sort_by=["models", "N"],
                     method=lambda models: names_dict[str(models)],
                     format=".pdf"
                     )

        plot_h_convergence(
            data_manager,
            path=config.subcell_paper_figures_path,
            name=f"HConvergence_{group}",
            folder=group,
            log="xy",
            models=models2plot,
            sort_by=["models", "h"],
            method=lambda models: names_dict[str(models)],
            format=".pdf",
            rateonly=rateonly
        )

        generic_plot(data_manager,
                     name=f"TimeComplexity_{group}",
                     path=config.subcell_paper_figures_path,
                     folder=group,
                     x="N", y="time", label="method", num_cells_per_dim=num_cells_per_dim,
                     plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--",
                                            palette={names_dict[k]: v for k, v in model_color.items()}),
                     log="xy",
                     time=lambda time_to_fit: time_to_fit,
                     N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     error=error,
                     models=models2plot,
                     method=lambda models: names_dict[str(models)],
                     format=".pdf"
                     )
        generic_plot(data_manager,
                     name=f"TimeComplexityPerCell_{group}",
                     path=config.subcell_paper_figures_path,
                     folder=group,
                     x="N", y="time", label="method", num_cells_per_dim=num_cells_per_dim,
                     plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--",
                                            palette={names_dict[k]: v for k, v in model_color.items()}),
                     log="xy",
                     time=curve_cells_fitting_times,
                     N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                     error=error,
                     models=models2plot,
                     method=lambda models: names_dict[str(models)],
                     format=".pdf"
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
                     method=lambda models: names_dict[str(models)],
                     format=".pdf"
                     )

    for group, models2plot in accepted_models.items():
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
            plot_again=True,
            num_cores=1,
            # trim=trim
        )

    print("CO2 consumption: ", data_manager.CO2kg)
    copy_main_script_version(__file__, data_manager.path)
