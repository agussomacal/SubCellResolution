import copy
import itertools
from pathlib import Path
from time import time
from typing import Union, Tuple, Type, List

import joblib
import numpy as np
from tqdm import tqdm

from PerplexityLab.miscellaneous import clean_str4saving, timeit, get_map_function
from lib.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
from lib.CellCreators.CellCreatorBase import get_rectangles_and_coords_to_calculate_flux
from lib.Curves.Curves import CurveBase
from lib.StencilCreators import Stencil

# from lib.Curves.CurveBase import CurveBase, calculate_integration_breakpoints_in_rectangle
# from lib.SubCellSchemes.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
# from lib.SubCellSchemes.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
# from lib.SubCellSchemes.CellCreators.CellCreatorBase import get_rectangles_and_coords_to_calculate_flux
# from lib.file_utils import load_joblib, save_joblib, erase_bad_symbols_before_saving
# from lib.performance_utils import timeit, get_map_function

CLASSIFICATION_PROBLEM = "Classification"
CURVE_PROBLEM = "Curve"
CELL_AVERAGES_PROBLEM = "CellAverages"
FLUX_PROBLEM = "Flux"
CURVE_CLASSIFICATION_PROBLEM = "Curve_classification"


def get_averages_from_curve_kernel(kernel_size: Tuple[int, ...], curve: CurveBase, center_cell_coords=None):
    kernel = np.zeros(kernel_size)
    center_cell_coords = np.array(kernel_size) // 2 if center_cell_coords is None else center_cell_coords
    for coords in map(np.array, itertools.product(*list(map(range, kernel_size)))):
        centered_coords = np.array(coords) - center_cell_coords - 0.5
        kernel[tuple(coords)] = curve.calculate_rectangle_average(
            x_limits=(centered_coords[0], centered_coords[0] + 1.0),
            y_limits=(centered_coords[1], centered_coords[1] + 1.0)
        )
    return kernel


def get_flux_from_curve_and_velocity(curve, center_cell_coords, velocity):
    # calculate flux
    next_coords, next_rectangles = get_rectangles_and_coords_to_calculate_flux(
        coords=np.array(center_cell_coords),
        velocity=velocity
    )
    flux = [curve.calculate_rectangle_average(
        x_limits=(rectangle[0][0], rectangle[1][0]),
        y_limits=(rectangle[0][1], rectangle[1][1])
    ) for rectangle in next_rectangles - center_cell_coords - 0.5]
    return flux


def is_central_cell_singular(kernel, center_cell_coords, minmax_val: Tuple):
    return minmax_val[1] > kernel[tuple(center_cell_coords)] > minmax_val[0]


def load_joblib(path: Union[str, Path]):
    with open(str(path), "r") as f:
        return joblib.load(f.name)


def save_joblib(path: Union[str, Path], variable):
    with open(str(path), "w") as f:
        joblib.dump(variable, f.name)


class DatasetsBaseManager:
    def __init__(self, path2data: Union[str, Path], curve_type: Type[CurveBase], N: int, kernel_size: Tuple[int, int],
                 min_val: float, max_val: float, velocity_range: Union[Tuple[Tuple, Tuple], List], recalculate=False,
                 workers=-1, transpose=False,
                 seed=42, reload_data=True, value_up_random=True):
        self.path2datafolder = Path(path2data)
        self.path2datafolder.mkdir(parents=True, exist_ok=True)
        self.workers = workers

        self.curve_type = curve_type
        self.N = N
        self.kernel_size = kernel_size
        self.transpose = transpose

        self.value_up_random = value_up_random

        self.recalculate = recalculate
        self.seed = seed

        # TODO: generalize method to receive a PDE instead of a velocity.
        if isinstance(velocity_range, tuple):
            assert np.all(np.diff(velocity_range, axis=0) >= 0), "max vel > min_vel should be."
            assert np.all(np.array(velocity_range) >= 0), "vel >= 0 should be."
            # assert np.all(np.array(velocity_range)[:, 0] > 0), "vel.x > 0 should be."
            assert np.all(np.array(velocity_range) <= 1), "max vel < 1 should be."
        else:
            assert np.all(np.max(np.abs(velocity_range)) <= 1), "max vel < 1 should be."

        self.velocity_range = velocity_range  # ((vx_min, vy_min), (vx_max, vy_max)) in fractions of discretisation
        self.center_cell_coords = np.array(kernel_size) // 2
        self.center = DatasetsBaseManager.get_center(kernel_size)

        self.minmax_val = (min_val, max_val)

        self.reload_data = reload_data
        self.__data = None

    def __len__(self):
        return self.N

    def __eq__(self, other):
        if isinstance(other, DatasetsBaseManager):
            return self.base_name == other.base_name
        else:
            raise Exception("Comparison between other than DatasetsManager not possible.")

    @staticmethod
    def get_center(kernel_size):
        return np.array(kernel_size) // 2 + 0.5

    @property
    def curve_types(self):
        return [self.curve_type]

    @property
    def T(self):
        new_self = copy.deepcopy(self)
        new_self.transpose = True
        return new_self

    # --------- file properties ---------- #
    @property
    def base_name(self):
        kernel_name = "_".join(map(str, self.kernel_size))
        return f"{self.curve_type.__name__}_k{kernel_name}_n{self.N}_min{self.minmax_val[0]}_max{self.minmax_val[1]}_" \
               f"v{clean_str4saving(str(self.velocity_range))}"

    @property
    def name4learning(self):
        """In case special data about the transformations has to be added."""
        return ""

    @property
    def data_filename(self):
        return f"Dataset4Learning_{self.base_name}"

    @property
    def path2data(self):
        return Path(Path.joinpath(self.path2datafolder,
                                  "{}.compressed".format(clean_str4saving(self.data_filename))))

    # --------- load ---------- #
    def load_dataset(self, n):
        if self.recalculate or not self.path2data.exists():
            with timeit("Generating data for {}".format(self.data_filename)):
                data = self.generate_dataset()
            self.save_dataset(data)
        else:
            if self.reload_data or self.__data is None:
                with timeit("Loading data from {}".format(self.path2data)):
                    data = load_joblib(self.path2data)
                if not self.reload_data:
                    self.__data = data
            else:
                data = self.__data
        # keep only the first n entries
        return {k: v[:min(n, self.N)] if isinstance(v, List) else v for k, v in data.items()}

    def transform_curve_data(self, *args):
        """
        The data may be saved always in the same format but transformed to get different Learning objectives.
        """
        return args

    def get_params_from_curve(self, curve):
        return curve.params

    def get_dataset4problem(self, type_of_problem, n=None, n_train=None):
        n = self.N if n is None else n
        data = self.load_dataset(n)
        input_data = data["kernel"]
        if self.transpose:
            input_data = np.swapaxes(input_data, -1, -2)

        if type_of_problem == CLASSIFICATION_PROBLEM:
            output_data = np.array(data["classification"])
        elif type_of_problem == FLUX_PROBLEM:
            velocity = np.array(data["velocity"])[:, ::-1] if self.transpose else np.array(data["velocity"])
            input_data = list(zip(input_data, velocity))
            output_data = np.array(data["flux"])
        elif type_of_problem in [CURVE_PROBLEM, CELL_AVERAGES_PROBLEM]:
            output_data = np.transpose(self.transform_curve_data(*np.transpose(data["curve"])))
            if type_of_problem == CELL_AVERAGES_PROBLEM:
                # invert input and output data
                input_data, output_data = output_data, np.reshape(input_data, (len(input_data), -1))
        elif type_of_problem == CURVE_CLASSIFICATION_PROBLEM:
            output_data = np.array(len(input_data) * [str(self.curve_type)])
        else:
            raise Exception("Type of problem {} not implemented.".format(type_of_problem))
        n_train = n if n_train is None else n_train
        return input_data[:n_train], output_data[:n_train], input_data[n_train:], output_data[n_train:]

    # --------- save ---------- #
    def save_dataset(self, data):
        save_joblib(self.path2data, data)

    # --------- generate data ---------- #
    def get_velocity(self):
        if isinstance(self.velocity_range, list):
            return np.array(self.velocity_range[np.random.choice(len(self.velocity_range))])
        else:
            return np.random.uniform(*self.velocity_range)

    def get_curve_data(self):
        """
        The center of the stencil is in the center of the kernel_size: //2+0.5
        -> go to DatasetsbaseManager generate_dataset
        """
        raise Exception("Not implemented.")

    def get_curve(self, curve_data, **kwargs):
        return self.curve_type(*curve_data, **kwargs)

    def get_value_up_down(self):
        index_up = np.random.randint(0, 2) if self.value_up_random else 0
        return {"value_up": self.minmax_val[index_up], "value_down": self.minmax_val[1 - index_up]}

    def generate_dataset(self):
        print(f"Generating data {self.base_name}...")
        np.random.seed(self.seed)

        def par_func(params):
            curve_data, velocity = params
            curve = self.get_curve(curve_data, **self.get_value_up_down())

            # Calculate averages
            # The coordinates must be so the origin is in the center of the central cell
            kernel = get_averages_from_curve_kernel(self.kernel_size, curve, self.center_cell_coords)
            classification = [is_central_cell_singular(kernel, self.center_cell_coords, self.minmax_val)]

            # calculate flux
            flux = get_flux_from_curve_and_velocity(curve, self.center_cell_coords, velocity)

            # TODO: hardcoded only one direction of flux
            return kernel, velocity, classification, np.ravel(curve_data), [flux[len(flux) == 3]]

        t0 = time()
        map_func = get_map_function(self.workers)
        params = list(zip([self.get_curve_data() for _ in range(self.N)],
                          [self.get_velocity() for _ in range(self.N)]))

        data = [line for line in tqdm(map_func(par_func, params), desc="Creating dataset.")]
        data = {k: v for k, v in zip(["kernel", "velocity", "classification", "curve", "flux"], zip(*data))}
        data["time_building_database"] = time() - t0
        return data

    # --------- create curve ---------- #
    def create_curve_from_params(self, curve_params, coords: CellCoords, independent_axis: int, value_up,
                                 value_down, stencil: Stencil) -> CurveBase:
        raise Exception("Not implemented.")


class DatasetConcatenator:
    def __init__(self, path2data: Union[str, Path], datasets: Union[DatasetsBaseManager, List[DatasetsBaseManager]],
                 curve_classification=None,
                 name=None):
        datasets = [datasets] if isinstance(datasets, DatasetsBaseManager) else datasets
        kernel_sizes = {dataset.kernel_size for dataset in datasets}
        assert len(kernel_sizes) == 1, f"Only concatenation between same kernel_size datasets but found {kernel_sizes}"
        self.name = name
        self.kernel_size = list(kernel_sizes).pop()
        self.center_cell_coords = np.array(self.kernel_size) // 2
        self.center = DatasetsBaseManager.get_center(self.kernel_size)

        self.path2datafolder = Path(path2data)
        self.path2datafolder.mkdir(parents=True, exist_ok=True)
        self.datasets = datasets
        self.curve_classification = curve_classification

    @property
    def curve_types(self):
        ct = []
        for dataset in self.datasets:
            if dataset.curve_type not in ct:
                ct.append(dataset.curve_type)
        return ct

    @property
    def curve_types_names(self):
        ct = []
        for dataset in self.datasets:
            if dataset.curve_type not in ct:
                ct.append(dataset.curve_type.__name__)
        return ct

    def __len__(self):
        return sum(map(len, self.datasets))

    def __eq__(self, other):
        if isinstance(other, DatasetConcatenator):
            return self.base_name == other.base_name
        else:
            raise Exception("Comparison between other than DatasetsManager not possible.")

    # --------- file properties ---------- #
    @property
    def base_name(self):
        if self.name is None:
            kernel_name = "_".join(map(str, self.kernel_size))
            curves_names = "_".join(self.curve_types_names)
            min_val = min([dataset.minmax_val[0] for dataset in self.datasets])
            max_val = min([dataset.minmax_val[1] for dataset in self.datasets])
            velocity_range = sorted(list(set(itertools.chain(*[dataset.velocity_range for dataset in self.datasets]))))

            return (f"N{len(self.datasets)}_"
                    f"{curves_names}_"
                    f"k{kernel_name}_"
                    f"n{len(self)}_"
                    f"min{min_val}_"
                    f"max{max_val}_"
                    f"v{clean_str4saving(str(velocity_range))}")
        else:
            return self.name

    @property
    def name4learning(self):
        """In case special data about the transformations has to be added."""
        return ""

    def get_dataset4problem(self, type_of_problem, n=None, n_train=None):
        # test:
        # tuple(map(lambda x: list(itertools.chain(*x)), zip(*[([i]*(i+3), [-i]*(i+3)) for i in [1, 2]])))
        input_train, output_train, input_test, output_test = tuple(map(lambda x: list(itertools.chain(*x)), zip(*[
            dataset.get_dataset4problem(type_of_problem, n=n, n_train=n_train) for
            dataset in self.datasets])))

        if type_of_problem == CURVE_CLASSIFICATION_PROBLEM:
            if self.curve_classification is not None:
                one_hot_encoding = np.eye(len(set(self.curve_classification)))
                one_hot_encoding = {str(dataset.curve_type): one_hot_encoding[membership] for dataset, membership in
                                    zip(self.datasets, self.curve_classification)}
            else:
                one_hot_encoding = dict(zip(map(str, self.curve_types), np.eye(len(self.curve_types))))
            output_train = [one_hot_encoding[v] for v in output_train]
            output_test = [one_hot_encoding[v] for v in output_test]
        return input_train, np.array(output_train), input_test, np.array(output_test)
