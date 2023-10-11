"""

References:
    EarlyStopping: https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    Deep Learning tutorial: https://deeplizard.com/learn/playlist/PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU
"""
import sys
import tempfile
from typing import Dict, Any

import keras.models
import matplotlib.pyplot as plt
import numpy as np
from keras.src.callbacks import EarlyStopping
from pathos.multiprocessing import cpu_count
from scikeras.wrappers import KerasRegressor
from sklearn.base import BaseEstimator
from tqdm.keras import TqdmCallback

from PerplexityLab.miscellaneous import get_map_function


class MLPRegressorKeras(KerasRegressor):

    def __init__(self,
                 hidden_layer_sizes, activation="relu", epochs=1000, restarts=1, max_time4fitting=np.Inf,
                 batch_size=None, loss='mse', optimizer='adam', lr=None,
                 validation_batch_size=None,
                 lr_lower_limit=1e-12, lr_upper_limit=1, n_epochs_without_improvement=100, random_state=42, workers=1,
                 verbose=0, warm_start=False, metrics=None, callbacks=None, validation_split=0.2, run_eagerly=False,
                 shuffle=True, **kwargs):
        # for other callbacks: https://keras.io/api/callbacks/#earlystopping
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True,
                           patience=n_epochs_without_improvement)

        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes

        self.history = None  # history of metrics while training.

        self.restarts = restarts
        self.max_time4fitting = max_time4fitting  # is in hours
        self.workers = workers if workers < cpu_count() else cpu_count() - 1

        # optimizer params
        self.lr = lr
        self.lr_lower_limit = lr_lower_limit
        self.lr_upper_limit = lr_upper_limit

        super().__init__(model=None,
                         build_fn=None,  # for backwards compatibility
                         warm_start=warm_start,
                         random_state=random_state,
                         optimizer=optimizer,
                         loss=loss,
                         metrics=metrics,
                         batch_size=batch_size,
                         validation_batch_size=validation_batch_size,
                         verbose=verbose,
                         callbacks=[es],
                         validation_split=validation_split,
                         shuffle=shuffle,
                         run_eagerly=run_eagerly,
                         epochs=epochs,
                         **kwargs)

    def _keras_build_fn(self, compile_kwargs: Dict[str, Any], *args, **kwargs):
        model = keras.Sequential()
        inp = keras.layers.Input(shape=(self.n_features_in_))
        model.add(inp)
        for hidden_layer_size in self.hidden_layer_sizes:
            layer = keras.layers.Dense(hidden_layer_size, activation=self.activation)
            model.add(layer)
        out = keras.layers.Dense(np.prod(self.y_ndim_))
        model.add(out)
        model.compile(loss=self.loss, optimizer=self.optimizer)
        return model
