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
from experiments.LearningMethods import flatter
from experiments.subcell_paper.global_params import SUB_CELL_DISCRETIZATION2BOUND_ERROR, OBERA_ITERS, \
    CCExtraWeight, runsinfo
from experiments.subcell_paper.obera_experiments import get_sub_cell_model, get_shape, plot_reconstruction
from experiments.subcell_paper.tools import calculate_averages_from_image, calculate_averages_from_curve, \
    singular_cells_mask
from lib.AuxiliaryStructures.Constants import REGULAR_CELL
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from lib.CellCreators.CurveCellCreators.ELVIRACellCreator import ELVIRACurveCellCreator
from lib.CellCreators.CurveCellCreators.LearningCurveCellCreator import LearningCurveCellCreator
from lib.CellCreators.CurveCellCreators.TaylorCurveCellCreator import TaylorCircleCurveCellCreator
from lib.CellCreators.CurveCellCreators.ValuesCurveCellCreator import ValuesCurveCellCreator, \
    ValuesLineConsistentCurveCellCreator
from lib.CellCreators.RegularCellCreator import MirrorCellCreator
from lib.CellIterators import iterate_all
from lib.CellOrientators import BaseOrientator, OrientPredefined
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.Curves.VanderCurves import CurveVandermondePolynomial
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import CURVE_PROBLEM
from lib.DataManagers.DatasetsManagers.DatasetsManagerLinearCurves import DatasetsManagerLinearCurves, ANGLE_OBJECTIVE
from lib.DataManagers.LearningMethodManager import LearningMethodManager
from lib.SmoothnessCalculators import indifferent
from lib.StencilCreators import StencilCreatorFixedShape, StencilCreatorAdaptive
from lib.SubCellReconstruction import SubCellReconstruction, ReconstructionErrorMeasureBase, CellCreatorPipeline, \
    reconstruct_by_factor


# N = int(1e6)
# workers = 10
# dataset_manager_3_8pi = DatasetsManagerLinearCurves(
#     velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
#     workers=workers, recalculate=False, learning_objective=ANGLE_OBJECTIVE, angle_limits=(-3 / 8, 3 / 8),
#     value_up_random=False
# )
#
# nnlm = LearningMethodManager(
#     dataset_manager=dataset_manager_3_8pi,
#     type_of_problem=CURVE_PROBLEM,
#     trainable_model=Pipeline(
#         [
#             ("Flatter", FunctionTransformer(flatter)),
#             ("NN", MLPRegressor(hidden_layer_sizes=(20, 20,), activation='relu', learning_rate_init=0.1,
#                                 learning_rate="adaptive", solver="lbfgs"))
#         ]
#     ),
#     refit=False, n2use=-1,
#     training_noise=1e-5, train_percentage=0.9
# )


def fit_model(sub_cell_model):
    def decorated_func(image, noise, image4error, sub_discretization2bound_error):
        # image = load_image(image)
        # avg_values = calculate_averages_from_image(image, num_cells_per_dim)
        np.random.seed(42)
        avg_values = image + np.random.uniform(-noise, noise, size=image.shape)

        model = sub_cell_model()
        print(f"Doing model: {model}")

        t0 = time.time()
        model.fit(average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"))
        t_fit = time.time() - t0

        t0 = time.time()
        reconstruction = model.reconstruct_by_factor(resolution_factor=sub_discretization2bound_error)
        t_reconstruct = time.time() - t0

        cells_error = list()
        for ix in zip(*np.where(singular_cells_mask(avg_values))):
            reconstruction_ix = reconstruct_by_factor(cells=model.cells, resolution=model.resolution,
                                                      resolution_factor=sub_discretization2bound_error,
                                                      cells2reconstruct=[ix])
            cells_error.append(np.mean(np.abs(image4error - reconstruction_ix)))

        return {
            "model": model,
            "time_to_fit": t_fit,
            "cells_error": np.array(cells_error),
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
                              orientators=[OrientPredefined(predefined_axis=0), OrientPredefined(predefined_axis=1)])


@fit_model
def elvira_w():
    return get_sub_cell_model(ELVIRACurveCellCreator, 1, "ELVIRA100", 0, CCExtraWeight, 2,
                              orientators=[OrientPredefined(predefined_axis=0), OrientPredefined(predefined_axis=1)])


@fit_model
def elvira_w_oriented():
    return get_sub_cell_model(ELVIRACurveCellCreator, 1, "ELVIRAGRAD", 0, CCExtraWeight, 2)


@fit_model
def elvira_go100_ref2():
    return get_sub_cell_model(ELVIRACurveCellCreator, 2, "ELVIRAGRAD", 0, CCExtraWeight, 2)


@fit_model
def linear_obera():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=0),
                natural_params=True), 1,
        "LinearOpt", OBERA_ITERS, 0, 2)


@fit_model
def linear_obera_w():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveVandermondePolynomial, degree=1, ccew=-1),
                natural_params=True), 1,
        "LinearOpt", OBERA_ITERS, CCExtraWeight, 2)


@fit_model
def linear_aero_w():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=CCExtraWeight)), 1,
        "LinearAvg100", 0, CCExtraWeight, 2,
        StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3,
                               center_weight=2.1))


@fit_model
def linear_aero_consistent():
    return get_sub_cell_model(partial(ValuesLineConsistentCurveCellCreator, ccew=CCExtraWeight), 1,
                              "LinearAvgConsistent", 0, CCExtraWeight, 1,
                              StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3,
                                                     center_weight=2.1))


@fit_model
def linear_aero():
    return get_sub_cell_model(
        partial(ValuesCurveCellCreator, vander_curve=partial(CurveAveragePolynomial, degree=1, ccew=0)), 1,
        "LinearAvg", 0, CCExtraWeight, 2,
        StencilCreatorAdaptive(smoothness_threshold=REGULAR_CELL, independent_dim_stencil_size=3,
                               center_weight=2.1))


# @fit_model
# def nn_linear():
#     return get_sub_cell_model(partial(LearningCurveCellCreator, nnlm), 1,
#                               "LinearNN", 0, CCExtraWeight, 2)


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name=f'LinearInterfaces',
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
        # piecewise_constant,
        elvira,
        # elvira_w_oriented,
        # linear_obera,
        linear_obera_w,
        # linear_aero,
        # linear_aero_w,
        # linear_aero_consistent,
        # nn_linear,

        # quadratic_obera_non_adaptive,
        # quadratic_obera,
        # quadratic_aero,

        # elvira_go100_ref2,
        # quadratic_aero_ref2,

        # circle_avg,
        # circle_vander_avg,
        recalculate=False
    )
    num_cells_per_dim = np.logspace(np.log10(20), np.log10(100), num=10, dtype=int).tolist()[:5]
    num_cores = 1
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

    # color_dict = {
    #     "piecewise_constant": "gray",
    #     "elvira": ,
    #     "elvira_w",
    #     "elvira_w_oriented",
    #     "linear_obera",
    #     "linear_aero",
    #     "linear_aero_w",
    #     "quadratic_obera_non_adaptive",
    #     "quadratic_obera",
    #     "quadratic_aero",
    #     "elvira_go100_ref2",
    #     "quadratic_aero_ref2",
    #     "circle_avg",
    #     "circle_vander_avg",
    # }
    #
    names_dict = {
        "piecewise_constant": "Piecewise Constant",
        "nn_linear": "NN Linear",
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
        # "elvira_go100_ref2",
        # "quadratic_aero_ref2",
        # "circle_avg",
        # "circle_vander_avg",
    }

    runsinfo.append_info(
        **{k.replace("_", "-"): v for k, v in names_dict.items()}
    )

    mse = lambda reconstruction, image4error: np.mean(((np.array(reconstruction) - image4error) ** 2).ravel())

    # -------------------- linear models -------------------- #
    linear_models = [
        "nn_linear",
        "elvira",
        "elvira_w",
        "elvira_w_oriented",
        "linear_obera",
        "linear_obera_w",
        "linear_aero",
        "linear_aero_w",
        "linear_aero_consistent"
    ]

    generic_plot(data_manager,
                 name="ReconstructionError",
                 x="method", y="cells_error", label="N", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.boxenplot, hue="method"),
                 log="y", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 models=linear_models,
                 method=lambda models: names_dict[str(models)]
                 )
    generic_plot(data_manager,
                 name="ReconstructionErrorBar",
                 x="method", y="error", label="method", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.violinplot,
                                        # hue="method"
                                        ),
                 log="y", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 models=linear_models,
                 error=lambda cells_error: cells_error,
                 method=lambda models: names_dict[str(models)]
                 )

    generic_plot(data_manager,
                 name="ConvergenceLinearModels",
                 x="N", y="mse", label="method", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=mse,
                 models=linear_models,
                 method=lambda models: names_dict[str(models)]
                 )

    generic_plot(data_manager,
                 name="TimeComplexityLinearModels",
                 x="N", y="time", label="method", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", time=lambda time_to_fit: time_to_fit, N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=mse,
                 models=linear_models,
                 method=lambda models: names_dict[str(models)]
                 )

    generic_plot(data_manager,
                 name="TimeComplexityMSELinearModels",
                 x="time", y="mse", label="method", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", time=lambda time_to_fit: time_to_fit, N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=mse,
                 models=linear_models,
                 method=lambda models: names_dict[str(models)]
                 )

    plot_reconstruction(
        data_manager,
        name="",
        folder='LinearMethodsReconstructionComparison',
        axes_by=['models'],
        models=linear_models,
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

    kvbf
    # -------------------- AERO models -------------------- #

    generic_plot(data_manager,
                 name="ConvergenceAERO",
                 x="N", y="mse", label="models", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=mse,
                 models=[
                     "piecewise_constant",
                     "elvira",
                     "elvira_w",
                     "elvira_w_oriented",
                     "linear_obera",
                     "linear_aero",
                     "linear_aero_w",
                     "quadratic_obera_non_adaptive",
                     "quadratic_obera",
                     "quadratic_aero",
                     "elvira_go100_ref2",
                     "quadratic_aero_ref2",
                     "circle_avg",
                     "circle_vander_avg",
                 ],
                 )

    generic_plot(data_manager,
                 name="ConvergenceModels",
                 x="N", y="mse", label="models", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=mse,
                 models=[
                     "piecewise_constant",
                     # "elvira",
                     # "elvira_w",
                     "elvira_w_oriented",
                     # "linear_obera",
                     # "linear_aero",
                     # "linear_aero_w",
                     "quadratic_obera_non_adaptive",
                     "quadratic_obera",
                     "quadratic_aero",
                     # "elvira_go100_ref2",
                     # "quadratic_aero_ref2",
                     "circle_avg",
                     # "circle_vander_avg",
                 ],
                 )

    generic_plot(data_manager,
                 name="TimeComplexityModels",
                 x="N", y="time", label="models", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", time=lambda time_to_fit: time_to_fit, N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=mse,
                 models=[
                     "piecewise_constant",
                     # "elvira",
                     # "elvira_w",
                     "elvira_w_oriented",
                     # "linear_obera",
                     # "linear_aero",
                     # "linear_aero_w",
                     "quadratic_obera_non_adaptive",
                     "quadratic_obera",
                     "quadratic_aero",
                     # "elvira_go100_ref2",
                     # "quadratic_aero_ref2",
                     "circle_avg",
                     # "circle_vander_avg",
                 ],
                 )
    generic_plot(data_manager,
                 name="TimeComplexityMSEModels",
                 x="time", y="mse", label="models", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", time=lambda time_to_fit: time_to_fit, N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=mse,
                 models=[
                     "piecewise_constant",
                     # "elvira",
                     # "elvira_w",
                     "elvira_w_oriented",
                     # "linear_obera",
                     # "linear_aero",
                     # "linear_aero_w",
                     "quadratic_obera_non_adaptive",
                     "quadratic_obera",
                     "quadratic_aero",
                     # "elvira_go100_ref2",
                     # "quadratic_aero_ref2",
                     "circle_avg",
                     # "circle_vander_avg",
                 ],
                 )

    generic_plot(data_manager,
                 name="ConvergenceRefinement",
                 x="N", y="mse", label="models", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=mse,
                 models=[
                     "piecewise_constant",
                     # "elvira",
                     # "elvira_w",
                     "elvira_w_oriented",
                     # "linear_obera",
                     # "linear_aero",
                     # "linear_aero_w",
                     # "quadratic_obera_non_adaptive",
                     # "quadratic_obera",
                     "quadratic_aero",
                     "elvira_go100_ref2",
                     "quadratic_aero_ref2",
                     # "circle_avg",
                     # "circle_vander_avg",
                 ],
                 )

    generic_plot(data_manager,
                 name="TimeComplexityMSERefinement",
                 x="time", y="mse", label="models", num_cells_per_dim=num_cells_per_dim,
                 mse=mse,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", time=lambda time_to_fit: time_to_fit, N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 models=[
                     "piecewise_constant",
                     # "elvira",
                     # "elvira_w",
                     "elvira_w_oriented",
                     # "linear_obera",
                     # "linear_aero",
                     # "linear_aero_w",
                     # "quadratic_obera_non_adaptive",
                     # "quadratic_obera",
                     "quadratic_aero",
                     "elvira_go100_ref2",
                     "quadratic_aero_ref2",
                     # "circle_avg",
                     # "circle_vander_avg",
                 ],
                 )

    generic_plot(data_manager,
                 name="TimeComplexityRefinement",
                 x="N", y="time", label="models", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", time=lambda time_to_fit: time_to_fit, N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 models=[
                     "piecewise_constant",
                     # "elvira",
                     # "elvira_w",
                     "elvira_w_oriented",
                     # "linear_obera",
                     # "linear_aero",
                     # "linear_aero_w",
                     # "quadratic_obera_non_adaptive",
                     # "quadratic_obera",
                     "quadratic_aero",
                     "elvira_go100_ref2",
                     "quadratic_aero_ref2",
                     # "circle_avg",
                     # "circle_vander_avg",
                 ],
                 )

    # plot_reconstruction(
    #     data_manager,
    #     name="",
    #     folder='ReconstructionComparison',
    #     axes_by=['models'],
    #     plot_by=['num_cells_per_dim'],
    #     axes_xy_proportions=(15, 15),
    #     difference=False,
    #     plot_curve=True,
    #     plot_curve_winner=False,
    #     plot_vh_classification=False,
    #     plot_singular_cells=False,
    #     plot_original_image=True,
    #     numbers_on=True,
    #     plot_again=True,
    #     num_cores=15,
    #     # trim=trim
    # )
    # plot_original_image(
    #     data_manager,
    #     folder='reconstruction',
    #     axes_by=[],
    #     plot_by=['image', 'models', 'num_cells_per_dim', 'refinement'],
    #     axes_xy_proportions=(15, 15),
    #     numbers_on=True
    # )

    print("CO2 consumption: ", data_manager.CO2kg)
    copy_main_script_version(__file__, data_manager.path)
