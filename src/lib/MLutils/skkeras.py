"""

References:
    EarlyStopping: https://machinelearningmastery.com/how-to-stop-training-deep-neural-networks-at-the-right-time-using-early-stopping/
    Deep Learning tutorial: https://deeplizard.com/learn/playlist/PLZbbT5o_s2xq7LwI2y8_QtvuXZedL6tQU
"""
import sys
import tempfile

import keras.models
import matplotlib.pyplot as plt
import numpy as np
from pathos.multiprocessing import cpu_count
from sklearn.base import BaseEstimator
from tqdm.keras import TqdmCallback

from PerplexityLab.miscellaneous import get_map_function


# TODO: not paralellizable yet


# ========= make sequential pickleable ========= #
def make_keras_picklable():
    """
    http://zachmoshe.com/2017/04/03/pickling-keras-models.html
    :return:
    """

    def __getstate__(self):
        model_str = ""
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            keras.models.save_model(self, fd.name, overwrite=True)
            model_str = fd.read()
        d = {'model_str': model_str}
        return d

    def __setstate__(self, state):
        with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            fd.write(state['model_str'])
            fd.flush()
            model = keras.models.load_model(fd.name)
        self.__dict__ = model.__dict__

    cls = keras.models.Model
    cls.__getstate__ = __getstate__
    cls.__setstate__ = __setstate__


# def unpack(model, training_config, weights):
#     from python.keras.layers import deserialize
#     from python.keras.saving import saving_utils
#
#     restored_model = deserialize(model)
#     if training_config is not None:
#         restored_model.compile(
#             **saving_utils.compile_args_from_training_config(
#                 training_config
#             )
#         )
#     restored_model.set_weights(weights)
#     return restored_model
#
#
# # Hotfix function
# def make_keras_picklable():
#     def __reduce__(self):
#         from python.keras.layers import serialize
#         from python.keras.saving import saving_utils
#         model_metadata = saving_utils.model_metadata(self)
#         training_config = model_metadata.get("training_config", None)
#         model = serialize(self)
#         weights = self.get_weights()
#         return (unpack, (model, training_config, weights))
#
#     from keras.models import Model
#     cls = Model
#     cls.__reduce__ = __reduce__


# Run the function
make_keras_picklable()


class SKKerasBase(BaseEstimator):
    def __init__(self, epochs=1000, restarts=1,
                 max_time4fitting=np.Inf, validation_size=0.2, batch_size=None, criterion='mse', solver='adam', lr=None,
                 lr_lower_limit=1e-12, lr_upper_limit=1, n_epochs_without_improvement=100, random_state=42, workers=1):
        super().__init__()

        self.random_state = random_state
        self.architecture = None

        self.history = None  # history of metrics while training.

        self.epochs = epochs
        self.restarts = restarts
        self.max_time4fitting = max_time4fitting  # is in hours
        self.workers = workers if workers < cpu_count() else cpu_count() - 1
        # from core.protobuf.config_pb2 import ConfigProto
        # from python.client.session import Session
        # from python.keras.backend import set_session
        # set_session(Session(config=ConfigProto(intra_op_parallelism_threads=self.workers,
        #                                        inter_op_parallelism_threads=self.workers)))

        # optimizer params
        self.criterion = criterion
        self.solver = solver
        self.lr = lr
        self.lr_lower_limit = lr_lower_limit
        self.lr_upper_limit = lr_upper_limit
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.n_epochs_without_improvement = n_epochs_without_improvement

        self.stats = {'epoch': [],
                      'learning rate': [],
                      'loss train': [],
                      'loss valid': []}

    @property
    def loss_valid(self):
        return self.history.history["val_loss"]

    def define_architecture(self, query, target):
        raise Exception("Not implemented.")

    def fit(self, query: np.ndarray, target: np.ndarray):
        self.batch_size = 1 if self.batch_size is None else self.batch_size
        self.batch_size = int(np.ceil(query.shape[0] * self.batch_size)) if self.batch_size <= 1 else self.batch_size

        # find learning rate
        if self.lr is None:
            try:
                from keras import optimizers
                optimizer = getattr(optimizers, self.solver)
            except:
                from tensorflow.keras import optimizers
                optimizer = getattr(optimizers, self.solver)

            model = self.define_architecture(query, target)
            model.compile(loss=self.criterion,
                          optimizer=optimizer(learning_rate=self.lr_lower_limit),
                          metrics=[])
            lrf = LearningRateFinder(model)
            self.lr = lrf.find([query, target], startLR=self.lr_lower_limit, endLR=self.lr_upper_limit,
                               stepsPerEpoch=np.ceil((len(query) / float(self.batch_size))),
                               batchSize=self.batch_size)
            # plot the loss for the various learning rates and save the
            # resulting plot to disk
            lrf.plot_loss()
            plt.show()
            del lrf
            # plt.savefig(config.LRFIND_PLOT_PATH)
            # self.find_lr(query, target)

        def train_in_paralel(restart):
            from keras.callbacks import EarlyStopping  # , ModelCheckpoint
            # from keras.models import load_model

            print("Restart number: {}".format(restart))
            model = self.define_architecture(query, target)
            model.compile(loss=self.criterion,
                          optimizer=optimizer(learning_rate=self.lr),
                          metrics=[])
            # for other callbacks: https://keras.io/api/callbacks/#earlystopping
            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, restore_best_weights=True,
                               patience=self.n_epochs_without_improvement)

            history = model.fit(query, target, epochs=self.epochs,
                                batch_size=self.batch_size, validation_split=self.validation_size,
                                callbacks=[es, TqdmCallback(verbose=0)], verbose=0)

            # with tempfile.NamedTemporaryFile(suffix='.hdf5', delete=True) as fd:
            #     # for other callbacks: https://keras.io/api/callbacks/#earlystopping
            #     mc = ModelCheckpoint(fd.name, monitor='val_loss', mode='min', save_best_only=True)
            #     es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,
            #                        patience=self.n_epochs_without_improvement)
            #
            #     history = model.fit(query, target, epochs=self.epochs,
            #                         batch_size=self.batch_size, validation_split=self.validation_size,
            #                         callbacks=[es, mc, TqdmCallback(verbose=0)], verbose=0)
            #
            #     model = load_model(fd.name)
            return model, history

        # Doesn't work the paralelization
        models = list(get_map_function(self.workers)(train_in_paralel, range(self.restarts)))
        # models = list(map(train_in_paralel, range(self.restarts)))
        print("Models min validation loss: ", list(map(lambda x: min(x[1].history["val_loss"]), models)))
        self.architecture, self.history = min(models, key=lambda x: min(x[1].history["val_loss"]))

    def predict(self, query: np.ndarray) -> np.array:
        return self.architecture.predict(query, verbose=0, workers=self.workers)

    def get_num_model_coefs(self):
        return self.architecture.count_params()


class SKKerasFNN(SKKerasBase):
    def __init__(self, hidden_layer_sizes, activation="relu", epochs=1000, restarts=1,
                 max_time4fitting=np.Inf, validation_size=0.2, batch_size=None, criterion='mse', solver='adam', lr=None,
                 lr_lower_limit=1e-12, lr_upper_limit=1, n_epochs_without_improvement=100, random_state=42, workers=1):
        """

        :param hidden_layer_sizes:
        :param activation: relu; sigmoid; softmax; softsign; tanh...
        :param epochs:
        :param restarts:
        :param max_time4fitting:
        :param validation_size:
        :param batch_size:
        :param criterion:
        :param solver:
        :param lr:
        :param lr_lower_limit:
        :param lr_upper_limit:
        :param n_epochs_without_improvement:
        :param random_state:
        :param workers:
        """
        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        super().__init__(epochs=epochs, validation_size=validation_size,
                         batch_size=batch_size, workers=workers,
                         restarts=restarts, max_time4fitting=max_time4fitting,
                         criterion=criterion, solver=solver,
                         lr=lr, lr_lower_limit=lr_lower_limit,
                         lr_upper_limit=lr_upper_limit,
                         n_epochs_without_improvement=n_epochs_without_improvement,
                         random_state=random_state)

    def define_architecture(self, query, target):
        from keras.layers import Dense
        from keras.models import Sequential

        model = Sequential()
        model.add(Dense(self.hidden_layer_sizes[0], input_shape=np.shape(query)[1:], activation=self.activation))
        for hidden_layer_size in self.hidden_layer_sizes[1:]:
            model.add(Dense(hidden_layer_size, activation=self.activation))
        model.add(Dense(np.prod(target.shape[1:])))
        return model


class LearningRateFinder:
    def __init__(self, model, stopFactor=4, beta=0.98):
        """
        Code from: https://pyimagesearch.com/2019/08/05/keras-learning-rate-finder/
        Based on paper: Cyclical Learning Rates for Training Neural Networks
        https://arxiv.org/abs/1506.01186
        :param model:
        :param stopFactor:
        :param beta:
        """

        # store the model, stop factor, and beta value (for computing
        # a smoothed, average loss)
        self.model = model
        self.stopFactor = stopFactor
        self.beta = beta
        # initialize our list of learning rates and losses,
        # respectively
        self.lrs = []
        self.losses = []
        # initialize our learning rate multiplier, average loss, best
        # loss found thus far, current batch number, and weights file
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def reset(self):
        # re-initialize all variables from our constructor
        self.lrs = []
        self.losses = []
        self.lrMult = 1
        self.avgLoss = 0
        self.bestLoss = 1e9
        self.batchNum = 0
        self.weightsFile = None

    def is_data_iter(self, data):
        # define the set of class types we will check for
        iterClasses = ["NumpyArrayIterator", "DirectoryIterator",
                       "DataFrameIterator", "Iterator", "Sequence"]
        # return whether our data is an iterator
        return data.__class__.__name__ in iterClasses

    def on_batch_end(self, batch, logs):
        from keras import backend as K

        # grab the current learning rate and add log it to the list of
        # learning rates that we've tried
        lr = K.get_value(self.model.optimizer.lr)
        self.lrs.append(lr)
        # grab the loss at the end of this batch, increment the total
        # number of batches processed, compute the average average
        # loss, smooth it, and update the losses list with the
        # smoothed value
        l = logs["loss"]
        self.batchNum += 1
        self.avgLoss = (self.beta * self.avgLoss) + ((1 - self.beta) * l)
        smooth = self.avgLoss / (1 - (self.beta ** self.batchNum))
        self.losses.append(smooth)
        # compute the maximum loss stopping factor value
        stopLoss = self.stopFactor * self.bestLoss
        # check to see whether the loss has grown too large
        if self.batchNum > 1 and smooth > stopLoss:
            # stop returning and return from the method
            self.model.stop_training = True
            return
        # check to see if the best loss should be updated
        if self.batchNum == 1 or smooth < self.bestLoss:
            self.bestLoss = smooth
        # increase the learning rate
        lr *= self.lrMult
        K.set_value(self.model.optimizer.lr, lr)

    def find(self, trainData, startLR, endLR, epochs=None,
             stepsPerEpoch=None, batchSize=32, sampleSize=2048,
             verbose=1):
        # reset our class-specific variables
        self.reset()
        # determine if we are using a data generator or not
        useGen = self.is_data_iter(trainData)
        # if we're using a generator and the steps per epoch is not
        # supplied, raise an error
        if useGen and stepsPerEpoch is None:
            msg = "Using generator without supplying stepsPerEpoch"
            raise Exception(msg)
        # if we're not using a generator then our entire dataset must
        # already be in memory
        elif not useGen:
            # grab the number of samples in the training data and
            # then derive the number of steps per epoch
            numSamples = len(trainData[0])
            stepsPerEpoch = np.ceil(numSamples / float(batchSize))
        # if no number of training epochs are supplied, compute the
        # training epochs based on a default sample size
        if epochs is None:
            epochs = int(np.ceil(sampleSize / float(stepsPerEpoch)))
        # compute the total number of batch updates that will take
        # place while we are attempting to find a good starting
        # learning rate
        numBatchUpdates = epochs * stepsPerEpoch
        # derive the learning rate multiplier based on the ending
        # learning rate, starting learning rate, and total number of
        # batch updates
        self.lrMult = (endLR / startLR) ** (1.0 / numBatchUpdates)
        # create a temporary file path for the model weights and
        # then save the weights (so we can reset the weights when we
        # are done)
        self.weightsFile = tempfile.mkstemp()[1]
        self.model.save_weights(self.weightsFile)
        # grab the *original* learning rate (so we can reset it
        # later), and then set the *starting* learning rate
        from keras.callbacks import LambdaCallback
        from keras import backend as K
        origLR = K.get_value(self.model.optimizer.lr)
        K.set_value(self.model.optimizer.lr, startLR)
        # construct a callback that will be called at the end of each
        # batch, enabling us to increase our learning rate as training
        # progresses
        callback = LambdaCallback(on_batch_end=lambda batch, logs:
        self.on_batch_end(batch, logs))
        # check to see if we are using a data iterator
        if useGen:
            self.model.fit(
                x=trainData,
                steps_per_epoch=stepsPerEpoch,
                epochs=epochs,
                callbacks=[callback, TqdmCallback(verbose=0)],
                verbose=0)
        # otherwise, our entire training data is already in memory
        else:
            # train our model using Keras' fit method
            self.model.fit(
                x=trainData[0], y=trainData[1],
                batch_size=batchSize,
                epochs=epochs,
                callbacks=[callback, TqdmCallback(verbose=0)],
                verbose=0)

        # restore the original model weights and set the optimal learning rate
        optim_lr = np.array(self.lrs)[np.argmin(self.losses)] / 10
        self.model.load_weights(self.weightsFile)
        K.set_value(self.model.optimizer.lr, optim_lr)
        return optim_lr

    def plot_loss(self, skipBegin=10, skipEnd=1, title=""):
        # grab the learning rate and losses values to plot
        lrs = self.lrs[skipBegin:-skipEnd]
        losses = self.losses[skipBegin:-skipEnd]
        # plot the learning rate vs. loss
        plt.plot(lrs, losses)
        plt.xscale("log")
        plt.xlabel("Learning Rate (Log Scale)")
        plt.ylabel("Loss")
        # if the title is not empty, add it to the plot
        if title != "":
            plt.title(title)
