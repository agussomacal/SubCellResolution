from functools import partial

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline, FunctionBlock
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version, ClassPartialInit
from PerplexityLab.visualization import generic_plot
from experiments.LearningMethods import skkeras_20x20_relu, skkeras_20x20_relu_noisy
from experiments.subcell_paper.ex_aero import fit_model, elvira_w_oriented, quadratic_aero
from experiments.subcell_paper.global_params import SUB_CELL_DISCRETIZATION2BOUND_ERROR, CCExtraWeight, VanderQuadratic
from experiments.subcell_paper.obera_experiments import get_sub_cell_model, get_shape, plot_reconstruction
from experiments.subcell_paper.tools import calculate_averages_from_curve
from lib.CellCreators.CurveCellCreators.LearningCurveCellCreator import LearningCurveCellCreator
from lib.Curves.VanderCurves import CurveVandermondePolynomial, CurveVanderCircle
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import CURVE_PROBLEM
from lib.DataManagers.DatasetsManagers.DatasetsManagerLinearCurves import DatasetsManagerLinearCurves, ANGLE_OBJECTIVE, \
    COS_SIN_OBJECTIVE
from lib.DataManagers.DatasetsManagers.DatasetsManagerVanderCurves import DatasetsManagerVanderCurves, POINTS_OBJECTIVE, \
    POINTS_SAMPLER_EQUISPACE, PARAMS_OBJECTIVE
from lib.DataManagers.LearningMethodManager import LearningMethodManager
from lib.StencilCreators import StencilCreatorFixedShape, StencilCreatorAdaptive

# SUB_CELL_DISCRETIZATION2BOUND_ERROR = 20
refit = False
recalculate = False
N = int(1e6)
workers = 15

print("dataset 1")
dataset_manager_3_8pi = DatasetsManagerLinearCurves(
    velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=15, recalculate=recalculate, learning_objective=ANGLE_OBJECTIVE, angle_limits=(-3 / 8, 3 / 8),
    value_up_random=False
)

print("dataset 2")
dataset_manager_cossin = DatasetsManagerLinearCurves(
    velocity_range=((0, 0), (0, 1)), path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=15, recalculate=recalculate, learning_objective=COS_SIN_OBJECTIVE,
    angle_limits=(-3 / 8, 3 / 8),
    value_up_random=False
)

print("dataset 3")
dataset_manager_vander = DatasetsManagerVanderCurves(
    curve_type=VanderQuadratic,
    velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(0.5, 0.5, 0.5), points_interval_size=1, value_up_random=False, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)

print("dataset 4")
dataset_manager_vander1 = DatasetsManagerVanderCurves(
    curve_type=VanderQuadratic,
    velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(1, 1, 1), points_interval_size=1, value_up_random=False, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)

print("dataset 5")
dataset_manager_vander7 = DatasetsManagerVanderCurves(
    curve_type=VanderQuadratic,
    velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 7), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(3.5, 1.5, 3.5), points_interval_size=3, value_up_random=False, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)

print("dataset 6")
dataset_manager_vander7params = DatasetsManagerVanderCurves(
    curve_type=VanderQuadratic,
    velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 7), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=PARAMS_OBJECTIVE,
    curve_position_radius=(3.5, 1.5, 3.5), points_interval_size=3, value_up_random=False, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)

print("dataset 7")
dataset_manager_vander7circle = DatasetsManagerVanderCurves(
    curve_type=CurveVanderCircle,
    velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 7), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(3.5, 1.5, 3.5), points_interval_size=3, value_up_random=False, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)

nnlm_cs = LearningMethodManager(
    dataset_manager=dataset_manager_cossin,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=skkeras_20x20_relu_noisy,
    refit=refit, n2use=-1,
    train_percentage=0.9
)

nnlm = LearningMethodManager(
    dataset_manager=dataset_manager_3_8pi,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=skkeras_20x20_relu_noisy,
    refit=refit, n2use=-1,
    train_percentage=0.9
)

nnlm4 = LearningMethodManager(
    dataset_manager=dataset_manager_3_8pi,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=skkeras_20x20_relu_noisy,
    refit=refit, n2use=1e4,
    train_percentage=0.9
)

nnlmq = LearningMethodManager(
    dataset_manager=dataset_manager_vander,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=skkeras_20x20_relu_noisy, refit=refit, n2use=-1,
    train_percentage=0.9
)

nnlmq1 = LearningMethodManager(
    dataset_manager=dataset_manager_vander1,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=skkeras_20x20_relu, refit=refit, n2use=-1,
    train_percentage=0.9
)

nnlmq7 = LearningMethodManager(
    dataset_manager=dataset_manager_vander7,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=skkeras_20x20_relu_noisy, refit=refit, n2use=-1,
    train_percentage=0.9
)

nnlmq7p = LearningMethodManager(
    dataset_manager=dataset_manager_vander7params,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=skkeras_20x20_relu_noisy, refit=refit, n2use=-1,
    train_percentage=0.9
)

nnlmc7 = LearningMethodManager(
    dataset_manager=dataset_manager_vander7circle,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=skkeras_20x20_relu_noisy, refit=refit, n2use=-1,
    train_percentage=0.9
)


def nn_linear_cs():
    return get_sub_cell_model(
        partial(LearningCurveCellCreator, learning_manager=nnlm_cs), 1, "NN_linear", 0, CCExtraWeight, 2)


def nn_linear():
    return get_sub_cell_model(
        partial(LearningCurveCellCreator, learning_manager=nnlm), 1, "NN_linear", 0, CCExtraWeight, 2)


def nn_linear4():
    return get_sub_cell_model(
        partial(LearningCurveCellCreator, learning_manager=nnlm4), 1, "NN_linear4", 0, CCExtraWeight, 2)


def nn_quadratic():
    return get_sub_cell_model(
        partial(LearningCurveCellCreator, learning_manager=nnlmq), 1, "NN_quadratic", 0, CCExtraWeight, 2)


def nn_quadratic_1():
    return get_sub_cell_model(
        partial(LearningCurveCellCreator, learning_manager=nnlmq1), 1, "NN_quadratic1", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorFixedShape(nnlmq1.dataset_manager.kernel_size))


def nn_quadratic3x7():
    return get_sub_cell_model(
        partial(LearningCurveCellCreator, learning_manager=nnlmq7), 1, "NN_quadratic3x7", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorFixedShape(nnlmq7.dataset_manager.kernel_size))


def nn_quadratic3x7params():
    return get_sub_cell_model(
        partial(LearningCurveCellCreator, learning_manager=nnlmq7p), 1, "NN_quadratic3x7params", 0, CCExtraWeight,
        2, stencil_creator=StencilCreatorFixedShape(nnlmq7p.dataset_manager.kernel_size))


def nn_quadratic3x7params_adapt():
    return get_sub_cell_model(
        partial(LearningCurveCellCreator, learning_manager=nnlmq7p), 1, "NN_quadratic3x7params", 0, CCExtraWeight,
        2, stencil_creator=StencilCreatorAdaptive(smoothness_threshold=0,
                                                  independent_dim_stencil_size=nnlmq7p.dataset_manager.kernel_size[0],
                                                  dependent_dim_size=nnlmq7p.dataset_manager.kernel_size[1]))


def nn_circle3x7():
    return get_sub_cell_model(
        partial(LearningCurveCellCreator, learning_manager=nnlmc7), 1, "NN_circle3x7", 0, CCExtraWeight, 2,
        stencil_creator=StencilCreatorFixedShape(nnlmc7.dataset_manager.kernel_size))


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='NN_beyond_lines',
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
        *list(map(fit_model,
                  [
                      # piecewise_constant,
                      # elvira,
                      # elvira_100,
                      elvira_w_oriented,
                      quadratic_aero,
                      # linear_obera,
                      # linear_avg,
                      # linear_avg_100,
                      nn_linear,
                      nn_linear_cs,
                      nn_linear4,
                      nn_quadratic,
                      nn_quadratic_1,
                      nn_quadratic3x7,
                      nn_quadratic3x7params,
                      nn_quadratic3x7params_adapt,
                      nn_circle3x7,
                  ])),
        recalculate=False
    )
    num_cells_per_dim = np.logspace(np.log10(20), np.log10(100), num=10, dtype=int).tolist()
    lab.execute(
        data_manager,
        num_cores=15,
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
    #     "elvira_100",
    #     "elvira_grad_oriented",
    #     "linear_obera",
    #     "linear_avg",
    #     "linear_avg_100",
    #     "quadratic_obera_non_adaptive",
    #     "quadratic_obera",
    #     "quadratic_avg",
    #     "elvira_go100_ref2",
    #     "quadratic_avg_ref2",
    #     "circle_avg",
    #     "circle_vander_avg",
    # }
    #
    # names_dict = {
    #     "piecewise_constant",
    #     "elvira",
    #     "elvira_100",
    #     "elvira_grad_oriented",
    #     "linear_obera",
    #     "linear_avg",
    #     "linear_avg_100",
    #     "quadratic_obera_non_adaptive",
    #     "quadratic_obera",
    #     "quadratic_avg",
    #     "elvira_go100_ref2",
    #     "quadratic_avg_ref2",
    #     "circle_avg",
    #     "circle_vander_avg",
    # }

    mse = lambda reconstruction, image4error: np.mean(((np.array(reconstruction) - image4error) ** 2).ravel())

    generic_plot(data_manager,
                 name="Convergence",
                 x="N", y="mse", label="models", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=mse,
                 # models=[
                 #     "piecewise_constant",
                 #     # "elvira",
                 #     # "elvira_100",
                 #     "elvira_grad_oriented",
                 #     "linear_obera",
                 #     "linear_avg",
                 #     "linear_avg_100",
                 #     "nn_linear",
                 # ],
                 )

    generic_plot(data_manager,
                 name="TimeComplexityLinearModels",
                 x="N", y="time", label="models", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", time=lambda time_to_fit: time_to_fit, N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=mse,
                 # models=[
                 #     "piecewise_constant",
                 #     # "elvira",
                 #     # "elvira_100",
                 #     "elvira_grad_oriented",
                 #     "linear_obera",
                 #     "linear_avg",
                 #     "linear_avg_100",
                 #     "nn_linear",
                 # ],
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
        num_cores=15,
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
    copy_main_script_version(__file__, data_manager.path)
