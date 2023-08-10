from functools import partial

import numpy as np
import seaborn as sns

import config
from PerplexityLab.DataManager import DataManager, JOBLIB
from PerplexityLab.LabPipeline import LabPipeline, FunctionBlock
from PerplexityLab.miscellaneous import NamedPartial, copy_main_script_version
from PerplexityLab.visualization import generic_plot
from experiments.LearningMethods import skkeras_20x20_relu
from experiments.subcell_paper.aero_experiments import fit_model, piecewise_constant, elvira, elvira_100, \
    elvira_grad_oriented, linear_obera, linear_avg, linear_avg_100
from experiments.subcell_paper.function_families import calculate_averages_from_curve
from experiments.subcell_paper.global_params import SUB_CELL_DISCRETIZATION2BOUND_ERROR, CCExtraWeight
from experiments.subcell_paper.obera_experiments import get_sub_cell_model, get_shape
from lib.CellCreators.CurveCellCreators.LearningCurveCellCreator import LearningCurveCellCreator
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import CURVE_PROBLEM
from lib.DataManagers.DatasetsManagers.DatasetsManagerLinearCurves import DatasetsManagerLinearCurves, ANGLE_OBJECTIVE
from lib.DataManagers.LearningMethodManager import LearningMethodManager

N = int(1e6)
dataset_manager_3_8pi = DatasetsManagerLinearCurves(
    velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=100, recalculate=False, learning_objective=ANGLE_OBJECTIVE, angle_limits=(-3 / 8, 3 / 8)
)


@fit_model
def nn_linear(metric):
    return get_sub_cell_model(
        partial(LearningCurveCellCreator, learning_manager=LearningMethodManager(
            dataset_manager=dataset_manager_3_8pi,
            type_of_problem=CURVE_PROBLEM, trainable_model=skkeras_20x20_relu, refit=False, n2use=-1,
            training_noise=0, train_percentage=0.9
        )), 1, "NN_linear", 0, CCExtraWeight, metric)


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='AEROandNN',
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
        elvira_100,
        elvira_grad_oriented,
        linear_obera,
        linear_avg,
        linear_avg_100,
        nn_linear,
        # quadratic_obera_non_adaptive,
        # quadratic_obera,
        # quadratic_avg,
        # elvira_go100_ref2,
        # quadratic_avg_ref2,
        # circle_avg,
        # circle_vander_avg,
        recalculate=False
    )
    num_cells_per_dim = np.logspace(np.log10(20), np.log10(100), num=10, dtype=int).tolist()
    lab.execute(
        data_manager,
        num_cores=1,
        forget=False,
        save_on_iteration=10,
        num_cells_per_dim=num_cells_per_dim,
        noise=[0],
        shape_name=[
            "Circle"
        ],
        sub_discretization2bound_error=[SUB_CELL_DISCRETIZATION2BOUND_ERROR],
        metric=[2]
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
                 models=[
                     "piecewise_constant",
                     "elvira",
                     "elvira_100",
                     "elvira_grad_oriented",
                     "linear_obera",
                     "linear_avg",
                     "linear_avg_100",
                     "quadratic_obera_non_adaptive",
                     "quadratic_obera",
                     "quadratic_avg",
                     "elvira_go100_ref2",
                     "quadratic_avg_ref2",
                     "circle_avg",
                     "circle_vander_avg",
                     "nn_linear",
                 ],
                 )

    generic_plot(data_manager,
                 name="TimeComplexityLinearModels",
                 x="N", y="time", label="models", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", time=lambda time_to_fit: time_to_fit, N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=mse,
                 models=[
                     # "piecewise_constant",
                     "elvira",
                     "elvira_100",
                     "elvira_grad_oriented",
                     "linear_avg",
                     "linear_avg_100",
                     "linear_obera",
                     "nn_linear",
                     # "quadratic_obera_non_adaptive",
                     # "quadratic_obera",
                     # "quadratic_avg",
                     # "elvira_go100_ref2",
                     # "quadratic_avg_ref2",
                     # "circle_avg",
                     # "circle_vander_avg",
                 ],
                 )

    generic_plot(data_manager,
                 name="TimeComplexityMSELinearModels",
                 x="time", y="mse", label="models", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", time=lambda time_to_fit: time_to_fit, N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 mse=mse,
                 models=[
                     # "piecewise_constant",
                     "elvira",
                     "elvira_100",
                     "elvira_grad_oriented",
                     "linear_avg",
                     "linear_avg_100",
                     "linear_obera",
                     "nn_linear",
                     # "quadratic_obera_non_adaptive",
                     # "quadratic_obera",
                     # "quadratic_avg",
                     # "elvira_go100_ref2",
                     # "quadratic_avg_ref2",
                     # "circle_avg",
                     # "circle_vander_avg",
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
                     # "elvira_100",
                     "elvira_grad_oriented",
                     # "linear_obera",
                     # "linear_avg",
                     # "linear_avg_100",
                     "quadratic_obera_non_adaptive",
                     "quadratic_obera",
                     "quadratic_avg",
                     # "elvira_go100_ref2",
                     # "quadratic_avg_ref2",
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
                     # "elvira_100",
                     "elvira_grad_oriented",
                     # "linear_obera",
                     # "linear_avg",
                     # "linear_avg_100",
                     "quadratic_obera_non_adaptive",
                     "quadratic_obera",
                     "quadratic_avg",
                     # "elvira_go100_ref2",
                     # "quadratic_avg_ref2",
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
                     # "elvira_100",
                     "elvira_grad_oriented",
                     # "linear_obera",
                     # "linear_avg",
                     # "linear_avg_100",
                     "quadratic_obera_non_adaptive",
                     "quadratic_obera",
                     "quadratic_avg",
                     # "elvira_go100_ref2",
                     # "quadratic_avg_ref2",
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
                     # "elvira_100",
                     "elvira_grad_oriented",
                     # "linear_obera",
                     # "linear_avg",
                     # "linear_avg_100",
                     # "quadratic_obera_non_adaptive",
                     # "quadratic_obera",
                     "quadratic_avg",
                     "elvira_go100_ref2",
                     "quadratic_avg_ref2",
                     # "circle_avg",
                     # "circle_vander_avg",
                 ],
                 )

    generic_plot(data_manager,
                 name="TimeComplexityMSERefinement",
                 x="time", y="mse", label="models", num_cells_per_dim=num_cells_per_dim,
                 plot_func=NamedPartial(sns.lineplot, marker="o", linestyle="--"),
                 log="xy", time=lambda time_to_fit: time_to_fit, N=lambda num_cells_per_dim: num_cells_per_dim ** 2,
                 models=[
                     "piecewise_constant",
                     # "elvira",
                     # "elvira_100",
                     "elvira_grad_oriented",
                     # "linear_obera",
                     # "linear_avg",
                     # "linear_avg_100",
                     # "quadratic_obera_non_adaptive",
                     # "quadratic_obera",
                     "quadratic_avg",
                     "elvira_go100_ref2",
                     "quadratic_avg_ref2",
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
                     # "elvira_100",
                     "elvira_grad_oriented",
                     # "linear_obera",
                     # "linear_avg",
                     # "linear_avg_100",
                     # "quadratic_obera_non_adaptive",
                     # "quadratic_obera",
                     "quadratic_avg",
                     "elvira_go100_ref2",
                     "quadratic_avg_ref2",
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
