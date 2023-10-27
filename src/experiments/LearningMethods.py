import numpy as np
import torch
# import torch
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from sklearn.tree import DecisionTreeRegressor

from lib.MLutils.scikit_keras import MLPRegressorKeras
from lib.MLutils.skkeras import SKKerasFNN, Keras2MLPRegressor
from lib.MLutils.sktorch import SKTorchFNN


def flatter(X):
    return np.array([np.concatenate([np.ravel(sub_item) for sub_item in x], axis=0) for x in X])


def real_modulus(x, p):
    return x - np.floor(x / p)


def periodize(X, periods):
    return np.array([X[:, i] if p is None else real_modulus(X[:, i], p) for i, p in enumerate(periods)]).T


def periodize_angle(X):
    X[:, 0] = real_modulus(X[:, 0], p=2 * np.pi)
    return X


class MLPRegressorT(MLPRegressor):
    def transform(self, X):
        return self.predict(X)


class Periodizer(BaseEstimator):
    def __init__(self, periods):
        self.periods = periods

    def fit(self, X, y=None):
        return self

    def predict(self, X):
        return np.array([X[:, i] if p is None else real_modulus(X[:, i], p) for i, p in enumerate(self.periods)]).T


LMNNFlatter = Pipeline(
    [
        ("Flatter", FunctionTransformer(flatter)),
        ("NN", MLPRegressor(hidden_layer_sizes=(20, 20,), activation='relu', learning_rate_init=0.1,
                            learning_rate="adaptive", solver="lbfgs"))
    ]
)
LMNNFlatterPeriodizer = Pipeline(
    [
        ("Flatter", FunctionTransformer(flatter)),
        ("NN", MLPRegressorT(hidden_layer_sizes=(20, 20,), activation='relu', learning_rate_init=0.1,
                             learning_rate="adaptive", solver="lbfgs")),
        # ("Flatter2", FunctionTransformer(flatter)),
        ("Periodize", Periodizer([2 * np.pi, None]))
    ]
)

# LMNNFlatter = Pipeline(
#     [
#         ("Flatter", FunctionTransformer(flatter)),
#         ("NN", DecisionTreeRegressor())
#     ]
# )

LMTreeFlatter = Pipeline(
    [
        ("Flatter", FunctionTransformer(flatter)),
        ("Tree", DecisionTreeRegressor())
    ]
)
LMRFFlatter = Pipeline(
    [
        ("Flatter", FunctionTransformer(flatter)),
        ("RF", RandomForestRegressor(n_estimators=50, n_jobs=-1))
    ]
)

skkeras_100x20_relu = Pipeline(
    [
        ("Flatter", FunctionTransformer(flatter)),
        ("SKKerasFNN100x20", SKKerasFNN(hidden_layer_sizes=(100, 20),
                                        epochs=100000, activation='relu', validation_size=0.1,
                                        restarts=5, max_time4fitting=np.Inf, workers=1,
                                        batch_size=5000, criterion="mse", solver="Adam",
                                        lr=None, lr_lower_limit=1e-12,
                                        lr_upper_limit=1,
                                        n_epochs_without_improvement=500,
                                        random_state=42))
    ]
)

skkeras_100x100_relu = Pipeline(
    [
        ("Flatter", FunctionTransformer(flatter)),
        ("SKKerasFNN100x100", Keras2MLPRegressor(SKKerasFNN(hidden_layer_sizes=(100, 100),
                                                            epochs=100000, activation='relu', validation_size=0.1,
                                                            restarts=5, max_time4fitting=np.Inf, workers=1,
                                                            batch_size=0.1, criterion="mse", solver="Adam",
                                                            lr=None, lr_lower_limit=1e-12,
                                                            lr_upper_limit=1, n_epochs_without_improvement=500,
                                                            random_state=42)))
    ]
)

skkeras_20x20_relu = Pipeline(
    [
        ("Flatter", FunctionTransformer(flatter)),
        ("SKKerasFNN20x20", Keras2MLPRegressor(SKKerasFNN(hidden_layer_sizes=(20, 20),
                                                          epochs=100000, activation='relu', validation_size=0.1,
                                                          restarts=1, max_time4fitting=np.Inf, workers=1,
                                                          batch_size=0.1, criterion="mse", solver="Adam",
                                                          lr=None, lr_lower_limit=1e-12,
                                                          lr_upper_limit=1, n_epochs_without_improvement=100,
                                                          random_state=42, train_noise=0)))
    ]
)

skkeras_20x20_relu_noisy = Pipeline(
    [
        ("Flatter", FunctionTransformer(flatter)),
        ("SKKerasFNN20x20", Keras2MLPRegressor(SKKerasFNN(hidden_layer_sizes=(20, 20),
                                                          epochs=100000, activation='relu', validation_size=0.1,
                                                          restarts=1, max_time4fitting=np.Inf, workers=1,
                                                          batch_size=0.1, criterion="mse", solver="Adam",
                                                          lr=None, lr_lower_limit=1e-12,
                                                          lr_upper_limit=1, n_epochs_without_improvement=100,
                                                          random_state=42, train_noise=1e-5)))
    ]
)

skpykeras_20x20_relu = Pipeline(
    [
        ("Flatter", FunctionTransformer(flatter)),
        ("SKskKerasFNN20x20", MLPRegressorKeras(hidden_layer_sizes=(20, 20),
                                                epochs=100000, activation='relu', validation_size=0.1,
                                                restarts=1, max_time4fitting=np.Inf, workers=1,
                                                batch_size=1000, criterion="mse", solver="adam",
                                                lr=None, lr_lower_limit=1e-12,
                                                lr_upper_limit=1, n_epochs_without_improvement=100,
                                                random_state=42))
    ]
)

sktorch_20x20_relu = Pipeline(
    [
        ("Flatter", FunctionTransformer(flatter)),
        ("SKTorchFNN20x20", SKTorchFNN(hidden_layer_sizes=(20, 20),
                                       epochs=100000, activation='relu', validation_size=0.1,
                                       restarts=1, max_time4fitting=np.Inf, workers=1,
                                       batch_size=0.1, criterion=torch.nn.MSELoss(), solver=torch.optim.Adam,
                                       lr=None, lr_lower_limit=1e-12,
                                       lr_upper_limit=1, n_epochs_without_improvement=100,
                                       random_state=42))
    ]
)
