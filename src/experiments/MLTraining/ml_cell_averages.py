import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline

import config
from experiments.MLTraining.ml_global_params import workers, N, recalculate
from experiments.VizReconstructionUtils import plot_cells
from experiments.subcell_paper.global_params import VanderQuadratic, AvgQuadratic
from experiments.subcell_paper.tools import get_evaluations2test_curve
from lib.Curves.CurveCircle import CurveSemiCircle, CircleParams
from lib.Curves.CurvePolynomial import CurvePolynomial, CurveLinearAngle
from lib.Curves.VanderCurves import CurveVanderCircle
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import get_averages_from_curve_kernel, \
    CELL_AVERAGES_PROBLEM
from lib.DataManagers.DatasetsManagers.DatasetsManagerLinearCurves import DatasetsManagerLinearCurves, ANGLE_OBJECTIVE, \
    SLOPE_OBJECTIVE
from lib.DataManagers.DatasetsManagers.DatasetsManagerVanderCurves import DatasetsManagerVanderCurves, POINTS_OBJECTIVE, \
    POINTS_SAMPLER_EQUISPACE, PARAMS_OBJECTIVE
from lib.DataManagers.LearningMethodManager import LearningMethodManager
from lib.MLutils.scikit_keras import SkKerasRegressor

value_up_random = False
refit = False

dataset_manager_lines = DatasetsManagerLinearCurves(
    velocity_range=[(0, 1 / 4), (1 / 4, 0), (0, -1 / 4), (-1 / 4, 0)],
    path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=workers, recalculate=False, learning_objective=SLOPE_OBJECTIVE, angle_limits=(-3 / 8, 3 / 8),
    value_up_random=value_up_random, curve_position_radius=1
)
dataset_manager_quadratics = DatasetsManagerVanderCurves(
    curve_type=VanderQuadratic,
    velocity_range=[(0, 1 / 4), (1 / 4, 0), (0, -1 / 4), (-1 / 4, 0)],
    path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=PARAMS_OBJECTIVE,
    curve_position_radius=(1.5, 0.5, 1.5), points_interval_size=1, value_up_random=value_up_random, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)
dataset_manager_quadratics_points = DatasetsManagerVanderCurves(
    curve_type=VanderQuadratic,
    velocity_range=[(0, 1 / 4), (1 / 4, 0), (0, -1 / 4), (-1 / 4, 0)],
    path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(1.5, 0.5, 1.5), points_interval_size=1, value_up_random=value_up_random, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)
dataset_manager_quadratics_avg = DatasetsManagerVanderCurves(
    curve_type=AvgQuadratic,
    velocity_range=[(0, 1 / 4), (1 / 4, 0), (0, -1 / 4), (-1 / 4, 0)],
    path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(1.5, 0.5, 1.5), points_interval_size=1, value_up_random=value_up_random, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)

dataset_manager_circle_3x7params = DatasetsManagerVanderCurves(
    curve_type=CurveVanderCircle,
    velocity_range=[(0, 1 / 4), (1 / 4, 0), (0, -1 / 4), (-1 / 4, 0)],
    path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=PARAMS_OBJECTIVE,
    curve_position_radius=(1.5, 0.5, 1.5), points_interval_size=1, value_up_random=value_up_random, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)
dataset_manager_circle_3x7points = DatasetsManagerVanderCurves(
    curve_type=CurveVanderCircle,
    velocity_range=[(0, 1 / 4), (1 / 4, 0), (0, -1 / 4), (-1 / 4, 0)],
    path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(1.5, 0.5, 1.5), points_interval_size=1, value_up_random=value_up_random, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)

regression_skkeras_20x20_relu_noisy = Pipeline(
    [
        ("SKKeras2020", SkKerasRegressor(hidden_layer_sizes=(20, 20,),
                                         epochs=100000, activation='logistic', validation_size=0.1,
                                         restarts=1,
                                         batch_size=0.1, criterion="mse", optimizer="Adam",
                                         lr=None, lr_lower_limit=1e-12,
                                         lr_upper_limit=1, n_epochs_without_improvement=100,
                                         train_noise=1e-5))
    ]
)

kernel_lines_ml_model = LearningMethodManager(
    dataset_manager=dataset_manager_lines,
    type_of_problem=CELL_AVERAGES_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=refit, n2use=-1,
    train_percentage=0.9
)

kernel_quadratics_ml_model = LearningMethodManager(
    dataset_manager=dataset_manager_quadratics,
    type_of_problem=CELL_AVERAGES_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=refit, n2use=-1,
    train_percentage=0.9
)
kernel_quadratics_points_ml_model = LearningMethodManager(
    dataset_manager=dataset_manager_quadratics_points,
    type_of_problem=CELL_AVERAGES_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=refit, n2use=-1,
    train_percentage=0.9
)
kernel_quadratics_avg_ml_model = LearningMethodManager(
    dataset_manager=dataset_manager_quadratics_avg,
    type_of_problem=CELL_AVERAGES_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=refit, n2use=-1,
    train_percentage=0.9
)
kernel_circles_ml_model_params = LearningMethodManager(
    dataset_manager=dataset_manager_circle_3x7params,
    type_of_problem=CELL_AVERAGES_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=refit, n2use=-1,
    train_percentage=0.9
)
kernel_circles_ml_model_points = LearningMethodManager(
    dataset_manager=dataset_manager_circle_3x7points,
    type_of_problem=CELL_AVERAGES_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=refit, n2use=-1,
    train_percentage=0.9
)
# vertex_ml_model = LearningMethodManager(
#     dataset_manager=dataset_manager_vertex,
#     type_of_problem=CURVE_PROBLEM,
#     trainable_model=regression_skkeras_20x20_relu_noisy,
#     refit=False, n2use=-1,
#     train_percentage=0.9
# )


if __name__ == "__main__":
    value_up = 0
    value_down = 1
    refinement = 5

    # curve = CurveVertexLinearAngleAngle(angle1=3 / 8 * np.pi, angle2=-3 / 8 * np.pi - np.pi, x0=0, y0=y0,
    #                                     value_up=value_up, value_down=value_down)

    # curve = CurveSemiCircle(params=CircleParams(x0=0, y0=-2.7, radius=3), value_up=value_up, value_down=value_down)
    # kernel_pred = kernel_circles_ml_model_params.predict_kernel(curve.params, reshape=True)
    # kernel_size = kernel_circles_ml_model_params.dataset_manager.kernel_size

    # curve = CurveVanderCircle(x_points=np.array([-1, 0, 1]), y_points=np.array([0.02, 0, -0.01]), value_up=value_up,
    #                           value_down=value_down)
    # kernel_pred = kernel_circles_ml_model_points.predict_kernel(curve.params, reshape=True)
    # kernel_size = kernel_circles_ml_model_points.dataset_manager.kernel_size
    #
    # curve = CurvePolynomial(polynomial=[0, 0.5, -0.25], value_up=value_up,
    #                         value_down=value_down)
    # kernel_pred = kernel_quadratics_ml_model.predict_kernel(curve.params, reshape=True)
    # kernel_size = kernel_quadratics_ml_model.dataset_manager.kernel_size

    y_points = np.array((5.993816730873693, 5.23852507146881, 4.479833111794201))-5
    # y_points = np.array([0.25, 0, -0.57])
    curve = VanderQuadratic(x_points=np.array([-1, 0, 1]), y_points=y_points, value_up=value_up,
                            value_down=value_down)
    kernel_pred = kernel_quadratics_points_ml_model.predict_kernel(curve.params, reshape=True)
    kernel_size = kernel_quadratics_points_ml_model.dataset_manager.kernel_size

    # curve = AvgQuadratic(x_points=np.array([-1, 0, 1]), y_points=np.array([0.02, 0, -0.01]), value_up=value_up,
    #                      value_down=value_down)
    # kernel_pred = kernel_quadratics_avg_ml_model.predict_kernel(curve.params, reshape=True)
    # kernel_size = kernel_quadratics_avg_ml_model.dataset_manager.kernel_size

    # curve = CurveLinearAngle(angle, y0, value_up, value_down)
    # kernel_pred = kernel_lines_ml_model.predict_kernel(curve.params[::-1], reshape=True)
    # curve = CurvePolynomial([0.23, np.tan(0.3)], value_up, value_down)
    # kernel_pred = kernel_lines_ml_model.predict_kernel(curve.params, reshape=True)
    # kernel_size = kernel_lines_ml_model.dataset_manager.kernel_size

    kernel = get_averages_from_curve_kernel(kernel_size, curve, center_cell_coords=None)
    u = get_evaluations2test_curve(curve, kernel_size, refinement=refinement)

    fig = plt.figure()
    ax = fig.add_gridspec(12, 4)
    ax1 = fig.add_subplot(ax[:4, :])
    ax1.set_title('True curve')
    ax2 = fig.add_subplot(ax[4:8, :])
    ax2.set_title('True Averages')
    ax3 = fig.add_subplot(ax[8:, :])
    ax3.set_title('Approx Averages')

    plot_cells(ax=ax1, colors=u, mesh_shape=np.array(kernel_size) * refinement, alpha=0.5,
               cmap="viridis",
               vmin=-1, vmax=1)
    sns.heatmap(kernel, annot=True, cmap="viridis", alpha=0.7, ax=ax2)
    sns.heatmap(kernel_pred, annot=True, cmap="viridis", alpha=0.7, ax=ax3)

    plt.tight_layout()
    plt.show()
