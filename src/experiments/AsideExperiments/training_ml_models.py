from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
import numpy as np

import config
from experiments.LearningMethods import flatter
from experiments.subcell_paper.global_params import VanderQuadratic
from lib.Curves.CurvePolynomial import CurveLinearAngle
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import DatasetConcatenator, CURVE_CLASSIFICATION_PROBLEM, \
    get_averages_from_curve_kernel
from lib.DataManagers.DatasetsManagers.DatasetsManagerLinearCurves import DatasetsManagerLinearCurves, ANGLE_OBJECTIVE
from lib.DataManagers.DatasetsManagers.DatasetsManagerVanderCurves import DatasetsManagerVanderCurves, POINTS_OBJECTIVE, \
    POINTS_SAMPLER_EQUISPACE
from lib.DataManagers.LearningMethodManager import LearningMethodManager
from lib.MLutils.scikit_keras import SkKerasClassifier

N = int(1e4)
recalculate = False
workers = 15

dataset_manager_3_8pi = DatasetsManagerLinearCurves(
    velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=15, recalculate=recalculate, learning_objective=ANGLE_OBJECTIVE, angle_limits=(-3 / 8, 3 / 8),
    value_up_random=False
)
dataset_manager_vander1 = DatasetsManagerVanderCurves(
    curve_type=VanderQuadratic,
    velocity_range=((0, 0), (1, 1)), path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(1, 1, 1), points_interval_size=1, value_up_random=False, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)

skkeras_20x20_relu_noisy = Pipeline(
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

nnlmkeras = LearningMethodManager(
    dataset_manager=DatasetConcatenator(config.data_path, dataset_manager_3_8pi, dataset_manager_vander1),
    type_of_problem=CURVE_CLASSIFICATION_PROBLEM,
    trainable_model=skkeras_20x20_relu_noisy,
    refit=False, n2use=-1,
    train_percentage=0.9
)

kernel_size = (3, 3)
angle = 0.15
y0 = 0.5
value_up = 0
value_down = 1

# curve = CurveLinearAngle(angle, y0, value_up, value_down, x_shift=0)
curve = VanderQuadratic(x_points=np.array([-1, 0, 1]), y_points=np.array([-1, 0, -1]), value_up=value_up,
                        value_down=value_down)
kernel = get_averages_from_curve_kernel(kernel_size, curve, center_cell_coords=None)
print(nnlmkeras.trainable_model.predict([kernel]))
print(nnlmkeras.trainable_model.predict_proba([kernel]))

curve = CurveLinearAngle(angle, y0, value_up, value_down, x_shift=0)
kernel = get_averages_from_curve_kernel(kernel_size, curve, center_cell_coords=None)
print(nnlmkeras.trainable_model.predict([kernel]))
print(nnlmkeras.trainable_model.predict_proba([kernel]))

nnlmkeras.predict_curve_type(kernel)
