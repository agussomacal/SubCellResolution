import copy
from pathlib import Path
from typing import Tuple, List, Union

import numpy as np
from sklearn.pipeline import Pipeline

from PerplexityLab.miscellaneous import clean_str4saving, timeit
from lib.CellCreators.CellCreatorBase import get_relative_next_coords_to_calculate_flux
from lib.Curves.Curves import CurveBase
from lib.DataManagers.DatasetsManagers.DatasetsBaseManager import DatasetsBaseManager, load_joblib, save_joblib, \
    DatasetConcatenator


def l1_error(predictions, true_values):
    return np.mean(np.abs(np.reshape(predictions, (len(true_values), -1)) - np.array(true_values)))


def linf_error(predictions, true_values):
    return np.max(np.abs(np.reshape(predictions, (len(true_values), -1)) - np.array(true_values)))


class LearningMethodManager:
    def __init__(self, dataset_manager: Union[DatasetsBaseManager, DatasetConcatenator], trainable_model: Pipeline,
                 type_of_problem: str, refit=False, n2use=-1, seed=42, train_percentage=1,
                 name=""):
        self.name = name
        self.dataset_manager = dataset_manager
        self.seed = seed
        self.refit = refit
        self.type_of_problem = type_of_problem
        self.n2use = len(self.dataset_manager) if n2use == -1 else n2use
        self.train_percentage = train_percentage
        self.n_train = int(self.n2use * train_percentage)

        self.trainable_model = copy.deepcopy(trainable_model)
        self.load_model()

    def free_memory(self):
        self.trainable_model = None

    # --------- file properties ---------- #
    @property
    def model_filename(self):
        # each step of the Pipeline composing the trainable model constitute the name to be saved
        trained_model_name = "_".join(list(zip(*self.trainable_model.steps))[0])
        return f"{self.name}{self.type_of_problem}_{trained_model_name}_{self.dataset_manager.base_name}{self.dataset_manager.name4learning}" \
               f"_n_train{self.n_train}"

    @property
    def path2model(self):
        return Path(Path.joinpath(self.dataset_manager.path2datafolder,
                                  "{}.compressed".format(clean_str4saving(self.model_filename))))

    # --------- load ---------- #
    def fit(self, input_data, output_data):
        with timeit("Training model: {}".format(self.model_filename)):
            self.trainable_model.fit(input_data, output_data)

    def load_model(self):
        if self.refit or not self.path2model.exists():
            train_x, train_y, test_x, test_y = self.dataset_manager.get_dataset4problem(
                type_of_problem=self.type_of_problem,
                n=self.n2use,
                n_train=self.n_train
            )
            with timeit("Training model: {}".format(self.model_filename)):
                self.trainable_model.fit(train_x, train_y)
            predictions_train = self.trainable_model.predict(train_x)
            print(f"L1 Loss in train set: {l1_error(predictions_train, train_y)}")
            print(f"Linf Loss in train set: {linf_error(predictions_train, train_y)}\n")
            if len(test_y) > 0:
                predictions_test = self.trainable_model.predict(test_x)
                print(f"L1 Loss in test set: {l1_error(predictions_test, test_y)}")
                print(f"Linf Loss in test set: {linf_error(predictions_test, test_y)}")
            save_joblib(self.path2model, self.trainable_model)
        else:
            with timeit("Loading trained model: {}".format(self.model_filename)):
                self.trainable_model = load_joblib(self.path2model)

    # -------------- predict --------------- #
    def predict_flux(self, kernel: np.ndarray, velocity: np.ndarray) -> (List[Tuple[int]], np.ndarray):
        next_flux = self.trainable_model.predict([[kernel, velocity]])[0]
        # # TODO: hardcoded no velocity
        # next_flux = np.array([0.0] * 3)
        # next_flux[1] = self.trainable_model.predict([[kernel]])[0]
        next_coords = get_relative_next_coords_to_calculate_flux(velocity)
        return next_coords, [next_flux] if isinstance(next_flux, float) else next_flux

    def predict_curve_params(self, kernel: np.ndarray) -> CurveBase:
        return self.trainable_model.predict([kernel])[0]

    def predict_classification(self, kernel: np.ndarray) -> int:
        return np.ravel(self.trainable_model.predict([kernel])[0])[0]

    def predict_curve_type(self, kernel: np.ndarray) -> int:
        return self.dataset_manager.curve_types[np.where(np.ravel(self.trainable_model.predict([kernel])) != 0)[0][0]]
