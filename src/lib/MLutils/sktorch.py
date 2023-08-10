"""

References:
    Batch normailzation in pytorch: https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd
    GPU vs CPU: https://analyticsindiamag.com/heres-why-gpus-win-over-cpus-when-it-comes-to-ml/
    Parallelize with Torch: https://towardsdatascience.com/speed-up-your-algorithms-part-1-pytorch-56d8a4ae7051
"""

import copy
import time
from collections import namedtuple
from typing import List

import cma
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.base import BaseEstimator
from sklearn.utils import shuffle
from tqdm import tqdm

from src.lib.performance_utils import get_workers

SIGMOID = LOGISTIC = 'sigmoid'
RELU = 'relu'
TANH = 'tanh'

BestModel = namedtuple('BestModel', 'weights valid_loss epoch')


class SKTorchBase(torch.nn.Module, BaseEstimator):
    def __init__(self, epochs=1000, restarts=1, max_time4fitting=np.Inf, n_epochs_without_improvement=100,
                 validation_size=0.2, batch_size=None, criterion=torch.nn.MSELoss(),
                 solver=torch.optim.LBFGS, other_solvers=(),
                 iterations_cma=1000, popsize=10, sigma_cma=1, ratio_grad_cma=None,
                 lr=None, lr_lower_limit=1e-12, lr_upper_limit=1,
                 random_state=42, workers=1, save_stats=False):
        super().__init__()

        # if torch.cuda.is_available():
        #     self.dev = "cuda:0"
        # else:
        self.dev = "cpu"

        self.random_state = random_state

        self.architecture = None
        self.input_shape = None
        self.output_shape = None

        # cma params
        self.ratio_grad_cma = epochs + 2 if ratio_grad_cma is None else ratio_grad_cma
        self.iterations_cma = iterations_cma
        self.popsize = popsize
        self.sigma_cma = sigma_cma

        # optimizer params
        self.criterion = criterion
        self.solver = solver
        self.solvers = (solver,) + other_solvers
        self.start_lr = lr
        self.lr = lr
        self.lr_lower_limit = lr_lower_limit
        self.lr_upper_limit = lr_upper_limit
        self.validation_size = validation_size
        self.batch_size = batch_size
        self.n_epochs_without_improvement = n_epochs_without_improvement

        self.epochs = epochs
        self.restarts = restarts * len(self.solvers)
        self.max_time4fitting = max_time4fitting  # is in hours
        self.workers = get_workers(workers)

        self.save_stats = save_stats
        self.stats = {'learning rate': [],
                      'loss train': [],
                      'loss valid': []}

    def define_architecture(self):
        raise Exception("Not implemented.")

    def forward(self, query):
        return self.architecture(query)

    def find_lr(self, query, target, batch_size, steps=30, plot=False):
        batch_ixes = np.random.choice(query.shape[0], size=batch_size)
        losses = []

        learning_rates = np.logspace(np.log10(self.lr_lower_limit), np.log10(self.lr_upper_limit), steps)
        weights = self.get_weights(self)
        for lr in tqdm(learning_rates, desc='Finding learning rate.'):
            self = self.set_weights(self, weights)
            self.train()
            optimizer = self.solver(self.parameters(), lr=lr)

            def closure():
                optimizer.zero_grad()  # Forward pass
                y_pred = self.double()(query[batch_ixes]).to(self.dev)
                loss = self.criterion(y_pred, target[batch_ixes])
                loss.backward()

                return loss

            optimizer.step(closure)

            with torch.no_grad():
                y_pred = self.double()(query[batch_ixes]).to(self.dev)
                losses.append(self.criterion(y_pred, target[batch_ixes]).detach().numpy())
        # for lr in tqdm(learning_rates, desc='Finding learning rate.'):
        #     self = self.set_weights(self, weights)
        #     self.train()
        #     optimizer = self.solver(self.parameters(), lr=lr)
        #
        #     def closure():
        #         optimizer.zero_grad()  # Forward pass
        #         y_pred = self.to(torch.double)(query[batch_ixes]).to(self.dev)
        #         loss = self.criterion(y_pred, target[batch_ixes])
        #         loss.backward()
        #
        #         return loss
        #
        #     optimizer.step(closure)
        #
        #     self.eval()
        #     y_pred = self.to(torch.double)(query[batch_ixes]).to(self.dev)
        #     losses.append(self.criterion(y_pred, target[batch_ixes]).detach().numpy())

        if plot:
            plt.semilogx(learning_rates, losses)
            plt.xlabel('learning rates')
            plt.ylabel('loss')
            plt.title('Finding optim learning rate')
            plt.show()
            plt.close()

        # why the 1/10?
        # https: // nanonets.com / blog / hyperparameter - optimization /
        lr = learning_rates[np.argmin(losses)] / 10
        print('Good learning rate to start with: {}'.format(lr))
        return lr

    @staticmethod
    def get_weights(m: torch.nn):
        return np.concatenate([p.detach().numpy().ravel() for p in m.parameters()])

    @staticmethod
    def set_weights(m, coefs):
        i = 0
        for p in m.parameters():
            i_next = i + np.prod(p.shape)
            p.data = torch.tensor(coefs[i:i_next], dtype=p.dtype).view(p.shape)
            i = i_next
        return m

    def fit_with_cma(self, query, target):
        with torch.no_grad():
            x0 = self.get_weights(self)

            def objective_function(coefs, model=copy.deepcopy(self)):
                model = self.set_weights(model, coefs)
                return self.criterion(model.float()(query), target).item()

            x, _ = cma.fmin2(objective_function=objective_function,
                             x0=x0, sigma0=self.sigma_cma,
                             options={'ftarget': -np.Inf, 'popsize': self.popsize,
                                      'maxfevals': self.popsize * self.iterations_cma})
        self = self.set_weights(self, x)

    @staticmethod
    def get_best_weights_until_now(best_models: List[BestModel]):
        return min(best_models, key=lambda t: t.valid_loss).weights

    def restart(self, restarts, epoch, query_train, target_train, batch_size, query_valid, target_valid):
        self.eval()
        self.define_architecture()
        self.architecture.to(self.dev)

        # find best lr to begin with.
        if self.start_lr is None:
            self.lr = self.find_lr(query_train, target_train, batch_size)
        else:
            self.lr = self.start_lr

        restarts -= 1
        self.solver = self.solvers[(self.restarts - restarts) % len(self.solvers)]
        print("now solver is: {}".format(self.solver))
        optimizer = self.solver(self.parameters(), lr=self.lr)
        if self.solver == torch.optim.SGD:
            for g in optimizer.param_groups:
                g['lr'] = self.lr

        best_model = BestModel(weights=self.get_weights(self),
                               valid_loss=self.criterion(query_valid, target_valid),
                               epoch=epoch)
        return restarts, optimizer, best_model

    def fit(self, query: np.ndarray, target: np.ndarray):
        raise Exception("Model broken, Solve bugs before using.")
        query = np.array(query)
        target = np.array(target)

        self.input_shape = np.shape(query)[1:]
        self.output_shape = np.shape(target)[1:]

        torch.manual_seed(self.random_state)
        n_epochs_without_train_improvement = self.n_epochs_without_improvement // 10

        # separate in train/valid.
        train_indices, valid_indices = self.get_train_valid_indexes(query)

        query_train = torch.from_numpy(query[train_indices]).to(self.dev).to(torch.double)
        target_train = torch.from_numpy(target[train_indices]).to(self.dev).to(torch.double)
        query_valid = torch.from_numpy(query[valid_indices]).to(self.dev).to(torch.double)
        target_valid = torch.from_numpy(target[valid_indices]).to(self.dev).to(torch.double)

        # TODO: do better by balancing, it may happen that the last batch has only one element.
        if self.batch_size is None:
            batch_size = query_train.size()[0]
        elif self.batch_size < 1:
            batch_size = int(query_train.size()[0] * self.batch_size)
        else:
            batch_size = self.batch_size
        batch_size = np.min((query_train.size()[0], batch_size))

        restarts, optimizer, best_model = self.restart(self.restarts + 1, 0, query_train, target_train, batch_size,
                                                       query_valid, target_valid)
        best_models = [best_model]

        t0 = time.time()
        for epoch in tqdm(range(1, self.epochs + 1)):
            # valid_loss = self.criterion(self.to(torch.double)(query_valid).to(self.dev), target_valid)
            valid_loss = self.get_valid_loss(query_valid, target_valid)
            self.add_epoch_stats(valid_loss, query_train, target_train, optimizer)
            print("\rEpoch {}: Validation loss = {}".format(epoch, valid_loss), end="")

            # before training step
            if time.time() - t0 > self.max_time4fitting * 3600:  # is in hours
                print('Stopping learning because reached maximum time: {}'.format(self.max_time4fitting))
                break

            if len(self.stats['loss valid']) > self.n_epochs_without_improvement:
                if epoch > best_models[-1].epoch + self.n_epochs_without_improvement:
                    print('\nValidation not improving, best until now: {}'.format(
                        np.min(list(map(lambda x: x.valid_loss, best_models)))))
                    if restarts > 1:
                        print("Restarting weights")
                        restarts, optimizer, best_model = self.restart(restarts, epoch, query_train, target_train,
                                                                       batch_size, query_valid, target_valid)
                        best_models.append(best_model)  # new best model to be optimized
                    elif restarts <= 1:
                        break

            # save info of best model
            if valid_loss < best_models[-1].valid_loss:
                self.eval()
                with torch.no_grad():
                    # save old weights in case of returning.
                    best_models[-1] = BestModel(weights=self.get_weights(self),
                                                valid_loss=valid_loss,
                                                epoch=epoch)

            # optimize
            if epoch % self.ratio_grad_cma < self.ratio_grad_cma - 1:
                # ----- gradient descent optimization -----
                # batches optimization
                self.train()
                permutation = torch.randperm(len(train_indices))
                for i in range(0, len(permutation), batch_size):
                    def closure():
                        batch_ixes = permutation[i: i + batch_size]

                        optimizer.zero_grad()  # Forward pass
                        loss_batch_train = self.criterion(self.to(torch.double)(query_train[batch_ixes]).to(self.dev),
                                                          target_train[batch_ixes])

                        loss_batch_train.backward()

                        # optimizer.zero_grad()  # Forward pass
                        #
                        # # valid_loss = self.criterion(self.to(torch.double)(query_valid).to(self.dev), target_valid)
                        # y_pred = self.to(torch.double)(query_train[batch_ixes]).to(self.dev)
                        # # y_pred = self(query_train[batch_ixes]).to(self.dev)
                        # loss_batch_train = self.criterion(y_pred, target_train[batch_ixes])
                        # # loss_batch_train.requires_grad = True
                        # # loss_batch_train.retain_grad()
                        # loss_batch_train.backward()
                        return loss_batch_train

                    optimizer.step(closure)

                # checking for restarts or learning rates
                # after training step
                # --- actualize learning rate if there is no improvement ---
                if self.solver == torch.optim.SGD:
                    if epoch > n_epochs_without_train_improvement and \
                            np.all(self.stats['learning rate'][-n_epochs_without_train_improvement:] ==
                                   self.stats['learning rate'][-1]) and \
                            np.sum(np.diff(self.stats['loss train'][-n_epochs_without_train_improvement:]) >= 0) \
                            >= n_epochs_without_train_improvement / 2:

                        self = self.set_weights(self, best_models[-1].weights)
                        for g in optimizer.param_groups:
                            g['lr'] = g['lr'] / 5
                        print(
                            'last loss train: {}'.format(self.stats['loss train'][-n_epochs_without_train_improvement]))
                        print('Reducing learning rate. {} -> {}'.format(self.stats['learning rate'][-1],
                                                                        np.mean(
                                                                            [g['lr'] for g in optimizer.param_groups])))

                    # --- restart learning rate if it is too small ---
                    if self.stats['learning rate'][-1] < self.lr_lower_limit:
                        print('Learning rate too small, restarting. Best until now: {}'.format(
                            np.min(list(map(lambda x: x.valid_loss, best_models)))))
                        restarts, optimizer, best_models[-1] = self.restart(restarts + 1, epoch, query_train,
                                                                            target_train,
                                                                            batch_size, query_valid, target_valid)

            else:
                # ----- do cma optimization -----
                print('Before cma: {}'.format(valid_loss))
                self.fit_with_cma(query_train, target_train)
                valid_loss = self.get_valid_loss(query_valid, target_valid)
                print('After cma: {}'.format(valid_loss))
                if valid_loss < best_models[-1].valid_loss:
                    _, optimizer, best_models[-1] = self.restart(restarts + 1, epoch, query_train, target_train,
                                                                 batch_size, query_valid, target_valid)
                else:
                    # reset old weights that where better
                    self = self.set_weights(self, best_models[-1].weights)

        # ----- set best model -----
        self = self.set_weights(self, self.get_best_weights_until_now(best_models))
        valid_loss = self.get_valid_loss(query_valid, target_valid)
        self.add_epoch_stats(valid_loss, query_train, target_train, optimizer)
        print('Final valid loss: {}'.format(valid_loss))

    def get_train_valid_indexes(self, query):
        shuffled_indices = shuffle(np.arange(query.shape[0]), random_state=self.random_state)
        pivot_point = int(len(shuffled_indices) * (1 - self.validation_size))
        train_indices = shuffled_indices[:pivot_point]
        valid_indices = shuffled_indices[pivot_point:]
        return train_indices, valid_indices

    def get_valid_loss(self, query_valid, target_valid):
        self.eval()
        with torch.no_grad():
            y_pred = self.to(torch.double)(query_valid).to(self.dev)
            valid_loss = self.criterion(y_pred, target_valid)
        return valid_loss

    def add_epoch_stats(self, valid_loss, query_train, target_train, optimizer):
        if self.save_stats:
            loss_train = self.criterion(self.to(torch.double)(query_train).to(self.dev), target_train)
            if self.solver == torch.optim.SGD:
                self.stats['learning rate'].append(np.mean([g['lr'] for g in optimizer.param_groups]))
            self.stats['loss train'].append(loss_train.item())
        self.stats['loss valid'].append(valid_loss.item())

    def predict(self, query: np.ndarray) -> np.array:
        self.eval()
        return self.to(torch.double)(torch.from_numpy(query).to(torch.double)).detach().numpy()

    def get_num_model_coefs(self):
        return len([w for w in self.get_weights(self)])


def get_activation_function(activation_name):
    # https://towardsdatascience.com/pytorch-layer-dimensions-what-sizes-should-they-be-and-why-4265a41e01fd
    if activation_name.lower() in [SIGMOID, LOGISTIC]:
        func = torch.nn.Sigmoid()
        gain = torch.nn.init.calculate_gain('sigmoid')
    elif activation_name.lower() in [RELU]:
        func = torch.nn.ReLU()
        gain = torch.nn.init.calculate_gain('relu')
    elif activation_name.lower() in [TANH]:
        func = torch.nn.Tanh()
        gain = torch.nn.init.calculate_gain('tanh')
    else:
        raise Exception('Activation function {} not implemented.'.format(activation_name))
    return func, gain


class SKTorchFNN(SKTorchBase):
    def __init__(self, hidden_layer_sizes, epochs=1000, activation='sigmoid', validation_size=0.2, restarts=1,
                 max_time4fitting=np.Inf, workers=1, batch_size=None, criterion=torch.nn.MSELoss(), other_solvers=(),
                 solver=torch.optim.LBFGS, iterations_cma=1000, popsize=10, sigma_cma=1, ratio_grad_cma=None, lr=None,
                 lr_lower_limit=1e-12, lr_upper_limit=1, n_epochs_without_improvement=100, random_state=42, dropout_p=0,
                 batch_normalization=False, save_stats=False):

        self.activation = activation
        self.hidden_layer_sizes = hidden_layer_sizes
        self.dropout_p = dropout_p
        self.batch_normalization = batch_normalization
        super().__init__(epochs=epochs, validation_size=validation_size,
                         batch_size=batch_size, workers=workers,
                         restarts=restarts, max_time4fitting=max_time4fitting,
                         criterion=criterion, solver=solver, other_solvers=other_solvers,
                         iterations_cma=iterations_cma, popsize=popsize, sigma_cma=sigma_cma,
                         ratio_grad_cma=ratio_grad_cma, lr=lr, lr_lower_limit=lr_lower_limit,
                         lr_upper_limit=lr_upper_limit, save_stats=save_stats,
                         n_epochs_without_improvement=n_epochs_without_improvement, random_state=random_state)

    def define_architecture(self):
        func, gain = get_activation_function(self.activation)

        prev_shape = list(self.input_shape) + list(self.hidden_layer_sizes)
        post_shape = list(self.hidden_layer_sizes) + list(self.output_shape)

        sequence = list()
        for i, (ishape, oshape) in enumerate(zip(prev_shape, post_shape)):

            linear = torch.nn.Linear(ishape, oshape, bias=True)
            torch.nn.init.xavier_uniform_(linear.weight, gain=gain)

            # --- Linear transformation (axons)
            sequence.append(linear)

            # --- Dropout
            if self.dropout_p > 0:
                sequence.append(torch.nn.Dropout(p=self.dropout_p))

            # --- applying batch norm
            if self.batch_normalization:
                sequence.append(torch.nn.BatchNorm1d(oshape))

            # --- only add activation function if it is not the outputlayer
            if i < len(prev_shape) - 1:
                sequence.append(func)

        self.architecture = torch.nn.Sequential(*sequence)
