import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import config
from experiments.LearningMethods import flatter
from experiments.subcell_paper.global_params import VanderQuadratic
from lib.Curves.CurvePolynomial import CurveLinearAngle
from lib.Curves.CurveVertex import CurveVertexLinearAngleAngle
from lib.Curves.VanderCurves import CurveVanderCircle
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import DatasetConcatenator, CURVE_CLASSIFICATION_PROBLEM, \
    get_averages_from_curve_kernel
from lib.DataManagers.DatasetsManagers.DatasetsManagerLinearCurves import DatasetsManagerLinearCurves, ANGLE_OBJECTIVE
from lib.DataManagers.DatasetsManagers.DatasetsManagerVanderCurves import DatasetsManagerVanderCurves, POINTS_OBJECTIVE, \
    POINTS_SAMPLER_EQUISPACE
from lib.DataManagers.DatasetsManagers.DatasetsManagerVertex import DatasetsManagerVertex
from lib.DataManagers.LearningMethodManager import LearningMethodManager
from lib.MLutils.scikit_keras import SkKerasClassifier

N = int(1e4)
recalculate = False
workers = 15

dataset_manager_lines = DatasetsManagerLinearCurves(
    # velocity_range=((0, 0), (1, 1)),
    velocity_range=[(0, 0.25), (0.25, 0)],
    path2data=config.data_path, N=N, kernel_size=(5, 5), min_val=0, max_val=1,
    workers=15, recalculate=recalculate, learning_objective=ANGLE_OBJECTIVE, angle_limits=(-3 / 8, 3 / 8),
    curve_position_radius=0.5,
    value_up_random=True
)
dataset_manager_quadratics = DatasetsManagerVanderCurves(
    curve_type=VanderQuadratic,
    # velocity_range=((0, 0), (1, 1)),
    velocity_range=[(0, 0.25), (0.25, 0)],
    path2data=config.data_path, N=N, kernel_size=(5, 5), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(1.5, 0.5, 1.5), points_interval_size=1, value_up_random=True, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)
dataset_manager_circles = DatasetsManagerVanderCurves(
    curve_type=CurveVanderCircle,
    # velocity_range=((0, 0), (1, 1)),
    velocity_range=[(0, 0.25), (0.25, 0)],
    path2data=config.data_path, N=N, kernel_size=(5, 5), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(1.5, 0.5, 1.5), points_interval_size=1, value_up_random=True, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)

dataset_manager_vertex = DatasetsManagerVertex(
    velocity_range=[(0, 1 / 4), (1 / 4, 0), (0, -1 / 4), (-1 / 4, 0)],
    path2data=config.data_path, N=N, kernel_size=(5, 5), min_val=0, max_val=1,
    angle1_limits=(-1, 1), angle_cone_limits=(-2 + 3 / 8, -3 / 8),
    workers=workers, recalculate=recalculate, curve_position_radius=1, value_up_random=False)

classiffication_skkeras_20x20_relu_noisy = Pipeline(
    [
        ("Flatter", FunctionTransformer(flatter)),
        ("SKKeras2020", SkKerasClassifier(hidden_layer_sizes=(20, 20),
                                          epochs=100000, activation='relu', validation_size=0.1,
                                          restarts=1,
                                          batch_size=0.1, criterion="binary_crossentropy", optimizer="Adam",
                                          lr=None, lr_lower_limit=1e-12,
                                          lr_upper_limit=1, n_epochs_without_improvement=100,
                                          train_noise=1e-5))
    ]
)

curve_classification_ml_model = LearningMethodManager(
    dataset_manager=DatasetConcatenator(config.data_path,
                                        datasets=[dataset_manager_lines, dataset_manager_lines.T,
                                                  dataset_manager_quadratics,
                                                  dataset_manager_quadratics.T,
                                                  dataset_manager_circles, dataset_manager_circles.T,
                                                  dataset_manager_vertex, dataset_manager_vertex.T],
                                        curve_classification=[0, 0, 1, 1, 1, 1, 2, 2],
                                        name="SauronDataset"
                                        ),
    type_of_problem=CURVE_CLASSIFICATION_PROBLEM,
    trainable_model=classiffication_skkeras_20x20_relu_noisy,
    refit=False, n2use=-1,
    train_percentage=0.9
)

if __name__ == "__main__":
    kernel_size = (5, 5)
    angle = 0.15
    y0 = 0.5
    value_up = 0
    value_down = 1

    # curve = CurveLinearAngle(angle, y0, value_up, value_down, x_shift=0)

    print("Vertex")
    curve = CurveVertexLinearAngleAngle(angle1=-1 / 8 * np.pi, angle2=2 / 8 * np.pi - np.pi, x0=0, y0=y0,
                                        value_up=value_up, value_down=value_down)
    kernel = get_averages_from_curve_kernel(kernel_size, curve, center_cell_coords=None)
    print(curve_classification_ml_model.trainable_model.predict([kernel]))
    print(curve_classification_ml_model.trainable_model.predict_proba([kernel]))
    print(curve_classification_ml_model.predict_curve_type_index(kernel))

    print("Line")
    curve = CurveLinearAngle(angle, y0, value_up, value_down, x_shift=0)
    kernel = get_averages_from_curve_kernel(kernel_size, curve, center_cell_coords=None)
    print(curve_classification_ml_model.trainable_model.predict([kernel]))
    print(curve_classification_ml_model.trainable_model.predict_proba([kernel]))
    print(curve_classification_ml_model.predict_curve_type_index(kernel))

    print("Quadratic")
    curve = VanderQuadratic(x_points=np.array([-1, 0, 1]), y_points=np.array([-1, 0, -1]), value_up=value_up,
                            value_down=value_down)
    kernel = get_averages_from_curve_kernel(kernel_size, curve, center_cell_coords=None)
    print(curve_classification_ml_model.trainable_model.predict([kernel]))
    print(curve_classification_ml_model.trainable_model.predict_proba([kernel]))
    print(curve_classification_ml_model.predict_curve_type_index(kernel))

    print("Circle")
    curve = CurveVanderCircle(x_points=np.array([-1, 0, 1]), y_points=np.array([-1, 0, -1]), value_up=value_up,
                              value_down=value_down)
    kernel = get_averages_from_curve_kernel(kernel_size, curve, center_cell_coords=None)
    print(curve_classification_ml_model.trainable_model.predict([kernel]))
    print(curve_classification_ml_model.trainable_model.predict_proba([kernel]))
    print(curve_classification_ml_model.predict_curve_type_index(kernel))
