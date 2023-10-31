import matplotlib.pylab as plt
import numpy as np
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

import config
from experiments.LearningMethods import flatter
from experiments.VizReconstructionUtils import plot_cells
from experiments.subcell_paper.global_params import VanderQuadratic
from experiments.subcell_paper.tools import get_evaluations2test_curve
from lib.Curves.CurvePolynomial import CurveLinearAngle
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import DatasetConcatenator, CURVE_CLASSIFICATION_PROBLEM, \
    get_averages_from_curve_kernel, FLUX_PROBLEM, get_flux_from_curve_and_velocity
from lib.DataManagers.DatasetsManagers.DatasetsManagerLinearCurves import DatasetsManagerLinearCurves, ANGLE_OBJECTIVE
from lib.DataManagers.DatasetsManagers.DatasetsManagerVanderCurves import DatasetsManagerVanderCurves, POINTS_OBJECTIVE, \
    POINTS_SAMPLER_EQUISPACE
from lib.DataManagers.LearningMethodManager import LearningMethodManager
from lib.MLutils.scikit_keras import SkKerasClassifier, SkKerasRegressor

N = int(1e4)
recalculate = False
workers = 15

dataset_manager_3_8pi = DatasetsManagerLinearCurves(
    # velocity_range=((0, 0), (1, 1)),
    velocity_range=[(0, 0.25), (0.25, 0)],
    path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=15, recalculate=recalculate, learning_objective=ANGLE_OBJECTIVE, angle_limits=(-3 / 8, 3 / 8),
    value_up_random=False
)
dataset_manager_vander1 = DatasetsManagerVanderCurves(
    curve_type=VanderQuadratic,
    # velocity_range=((0, 0), (1, 1)),
    velocity_range=[(0, 0.25), (0.25, 0)],
    path2data=config.data_path, N=N, kernel_size=(3, 3), min_val=0, max_val=1,
    workers=workers, recalculate=recalculate, learning_objective=POINTS_OBJECTIVE,
    curve_position_radius=(1, 1, 1), points_interval_size=1, value_up_random=False, num_points=3,
    points_sampler=POINTS_SAMPLER_EQUISPACE,
)

# ------------------- CURVE CLASSIFICATION PROBLEM ------------------- #
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

nnlmkeras = LearningMethodManager(
    dataset_manager=DatasetConcatenator(config.data_path, dataset_manager_3_8pi, dataset_manager_vander1),
    type_of_problem=CURVE_CLASSIFICATION_PROBLEM,
    trainable_model=classiffication_skkeras_20x20_relu_noisy,
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

# ------------------- Flux PROBLEM ------------------- #
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

lines = LearningMethodManager(
    dataset_manager=dataset_manager_3_8pi,
    type_of_problem=FLUX_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=False, n2use=-1,
    train_percentage=0.9
)
parables = LearningMethodManager(
    dataset_manager=dataset_manager_vander1,
    type_of_problem=FLUX_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=False, n2use=-1,
    train_percentage=0.9
)
linpar = LearningMethodManager(
    dataset_manager=DatasetConcatenator(config.data_path, dataset_manager_3_8pi, dataset_manager_vander1),
    type_of_problem=FLUX_PROBLEM,
    trainable_model=regression_skkeras_20x20_relu_noisy,
    refit=False, n2use=-1,
    train_percentage=0.9
)

kernel_size = (3, 3)
angle = 0.0
y0 = 0.5
velocity = (0.25, 0)
# velocity = (0.0, 0.25)
value_up = 0
value_down = 1
refinement = 5

curve = VanderQuadratic(x_points=np.array([-1, 0, 1]), y_points=np.array([-0.3, 0, -0.1]), value_up=value_up,
                        value_down=value_down)
# curve = CurveLinearAngle(angle, y0, value_up, value_down, x_shift=0)
kernel = get_averages_from_curve_kernel(kernel_size, curve, center_cell_coords=None)
u = get_evaluations2test_curve(curve, kernel_size, refinement=refinement)

data2plot = dict()
data2plot["Flux"] = \
    get_flux_from_curve_and_velocity(curve, center_cell_coords=np.array(kernel_size) // 2, velocity=velocity)[0]
_, data2plot["Lines"] = lines.predict_flux(kernel, velocity)
_, data2plot["Quadratics"] = parables.predict_flux(kernel, velocity)
_, data2plot["Lines and Quadratics"] = linpar.predict_flux(kernel, velocity)

fig = plt.figure()
ax = fig.add_gridspec(6, 5)
ax1 = fig.add_subplot(ax[:3, 0:3])
ax1.set_title('Averages')
ax2 = fig.add_subplot(ax[:3, 3:])
ax2.set_title('True curve')
ax3 = fig.add_subplot(ax[3:, :])
ax3.set_title('Fluxes')

sns.heatmap(kernel, annot=True, cmap="viridis", alpha=0.7, ax=ax1)
plot_cells(ax=ax2, colors=u, mesh_shape=np.array(kernel_size) * refinement, alpha=0.5,
           cmap="viridis",
           vmin=-1, vmax=1)
ax3.barh(list(data2plot.keys()), list(data2plot.values()))
plt.tight_layout()
plt.show()
