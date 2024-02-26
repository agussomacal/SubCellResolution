import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import config
from experiments.OtherExperiments.MLTraining.LearningMethods import flatter
from experiments.OtherExperiments.MLTraining.ml_global_params import workers, recalculate, N
from experiments.VizReconstructionUtils import plot_cells
from experiments.global_params import VanderQuadratic
from experiments.tools import get_evaluations2test_curve
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.Curves.CurveVertex import CurveVertexLinearAngleAngle
from lib.Curves.VanderCurves import CurveVanderCircle
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import get_averages_from_curve_kernel, \
    CURVE_PROBLEM
from lib.DataManagers.DatasetsManagers.DatasetsManagerLinearCurves import DatasetsManagerLinearCurves, ANGLE_OBJECTIVE
from lib.DataManagers.DatasetsManagers.DatasetsManagerVanderCurves import DatasetsManagerVanderCurves, POINTS_OBJECTIVE, \
    POINTS_SAMPLER_EQUISPACE, PARAMS_OBJECTIVE
from lib.DataManagers.LearningMethodManager import LearningMethodManager
from lib.MLutils.scikit_keras import SkKerasRegressor

# False because there is the unidimensional mapping that makes 1 below 0 above
value_up_random = False

dataset_manager_lines = DatasetsManagerLinearCurves(
    velocity_range=[(0, 1 / 4), (1 / 4, 0), (0, -1 / 4), (-1 / 4, 0)],
    path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=workers, recalculate=False, learning_objective=ANGLE_OBJECTIVE, angle_limits=(-3 / 8, 3 / 8),
    value_up_random=value_up_random, curve_position_radius=1
)
dataset_manager_quadratics = DatasetsManagerVanderCurves(
    curve_type=VanderQuadratic,
    velocity_range=[(0, 1 / 4), (1 / 4, 0), (0, -1 / 4), (-1 / 4, 0)],
    path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(1.5, 0.5, 1.5), points_interval_size=1, value_up_random=value_up_random, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)
dataset_manager_quadratics_7 = DatasetsManagerVanderCurves(
    curve_type=VanderQuadratic,
    velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 7), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(3.5, 1.5, 3.5), points_interval_size=3, value_up_random=value_up_random, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)
dataset_manager_quadratics_7params = DatasetsManagerVanderCurves(
    curve_type=VanderQuadratic,
    velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 7), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=PARAMS_OBJECTIVE,
    curve_position_radius=(3.5, 1.5, 3.5), points_interval_size=3, value_up_random=value_up_random, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)
dataset_manager_circle_7 = DatasetsManagerVanderCurves(
    curve_type=CurveVanderCircle,
    velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 7), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(3.5, 1.5, 3.5), points_interval_size=3, value_up_random=value_up_random, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)

regression_skkeras_20x20_relu_noisy = Pipeline(
    [
        ("Flatter", FunctionTransformer(flatter)),
        ("SKKeras2020", SkKerasRegressor(hidden_layer_sizes=(20, 20),
                                         epochs=100000, activation='relu', validation_size=0.1,
                                         restarts=1,
                                         batch_size=0.1, criterion="mse", optimizer="Adam",
                                         lr=None, lr_lower_limit=1e-12,
                                         lr_upper_limit=1, n_epochs_without_improvement=100,
                                         train_noise=1e-5))
    ]
)

lines_ml_model = LearningMethodManager(
    dataset_manager=dataset_manager_lines,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=False, n2use=-1,
    train_percentage=0.9
)
quadratics_ml_model = LearningMethodManager(
    dataset_manager=dataset_manager_quadratics,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=False, n2use=-1,
    train_percentage=0.9
)
quadratics7_points_ml_model = LearningMethodManager(
    dataset_manager=dataset_manager_quadratics_7,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=False, n2use=-1,
    train_percentage=0.9
)
quadratics7_params_ml_model = LearningMethodManager(
    dataset_manager=dataset_manager_quadratics_7params,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=False, n2use=-1,
    train_percentage=0.9
)
circles7_ml_model = LearningMethodManager(
    dataset_manager=dataset_manager_circle_7,
    type_of_problem=CURVE_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=False, n2use=-1,
    train_percentage=0.9
)


# vertex_ml_model = LearningMethodManager(
#     dataset_manager=dataset_manager_vertex,
#     type_of_problem=CURVE_PROBLEM,
#     trainable_model=regression_skkeras_20x20_relu_noisy,
#     refit=False, n2use=-1,
#     train_percentage=0.9
# )

def get_evaluations2test_ml_curve(curve_params, data_manager, kernel_size, value_up, value_down,
                                  center_cell_coords=None) -> np.ndarray:
    center_cell_coords = np.array(kernel_size) // 2 if center_cell_coords is None else center_cell_coords
    curve = data_manager.create_curve_from_params(
        curve_params=curve_params,
        coords=CellCoords(center_cell_coords),
        independent_axis=0,
        value_up=value_up,
        value_down=value_down,
        stencil=None
    )
    return get_evaluations2test_curve(curve, kernel_size, refinement=refinement)


if __name__ == "__main__":
    kernel_size = (3, 3)
    angle = 0.0
    y0 = 0.23
    value_up = 0
    value_down = 1
    refinement = 5

    curve = CurveVertexLinearAngleAngle(angle1=3 / 8 * np.pi, angle2=-3 / 8 * np.pi - np.pi, x0=0, y0=y0,
                                        value_up=value_up, value_down=value_down)
    # curve = VanderQuadratic(x_points=np.array([-1, 0, 1]), y_points=np.array([-0.3, 0, -0.1]), value_up=value_up,
    #                         value_down=value_down)
    # curve = CurveLinearAngle(angle, y0, value_up, value_down)
    kernel = get_averages_from_curve_kernel(kernel_size, curve, center_cell_coords=None)
    u = get_evaluations2test_curve(curve, kernel_size, refinement=refinement)

    data2plot = dict()
    data2plot["Params"] = curve.params
    data2plot["Lines"] = lines_ml_model.predict_curve_params(kernel)
    data2plot["Quadratics"] = quadratics_ml_model.predict_curve_params(kernel)

    u_lines = get_evaluations2test_ml_curve(data2plot["Lines"], lines_ml_model.dataset_manager, kernel_size, value_up,
                                            value_down)
    u_quadratics = get_evaluations2test_ml_curve(data2plot["Quadratics"], quadratics_ml_model.dataset_manager,
                                                 kernel_size, value_up, value_down)

    fig = plt.figure()
    ax = fig.add_gridspec(8, 8)
    ax1 = fig.add_subplot(ax[:3, :3])
    ax1.set_title('Averages')
    ax2 = fig.add_subplot(ax[:3, 5:])
    ax2.set_title('True curve')
    ax3 = fig.add_subplot(ax[5:, :3])
    ax3.set_title('Line approx')
    ax4 = fig.add_subplot(ax[5:, 5:])
    ax4.set_title('Quadratic approx')

    sns.heatmap(kernel, annot=True, cmap="viridis", alpha=0.7, ax=ax1)
    plot_cells(ax=ax2, colors=u, mesh_shape=np.array(kernel_size) * refinement, alpha=0.5,
               cmap="viridis",
               vmin=-1, vmax=1)

    plot_cells(ax=ax3, colors=u_lines, mesh_shape=np.array(kernel_size) * refinement, alpha=0.5,
               cmap="viridis",
               vmin=-1, vmax=1)

    plot_cells(ax=ax4, colors=u_quadratics, mesh_shape=np.array(kernel_size) * refinement, alpha=0.5,
               cmap="viridis",
               vmin=-1, vmax=1)
    plt.tight_layout()
    plt.show()
