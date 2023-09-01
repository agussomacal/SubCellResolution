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

# from lib.Curves.CurveBase import CurveBase, calculate_integration_breakpoints_in_rectangle
# from lib.SubCellSchemes.AuxiliaryStructures.Constants import REGULAR_CELL, CURVE_CELL
# from lib.SubCellSchemes.AuxiliaryStructures.IndexingAuxiliaryFunctions import CellCoords
# from lib.SubCellSchemes.CellCreators.CellCreatorBase import get_rectangles_and_coords_to_calculate_flux
# from lib.file_utils import load_joblib, save_joblib, erase_bad_symbols_before_saving
# from lib.performance_utils import timeit, get_map_function

CLASSIFICATION_PROBLEM = "Classification"
CURVE_PROBLEM = "Curve"
FLUX_PROBLEM = "Flux"


def evaluate_function_in_rectangle(function, rectangle: np.ndarray, return_points=False):
    p1 = rectangle[0].copy()
    p2 = rectangle[0][0], rectangle[1][1]
    p3 = rectangle[1][0], rectangle[0][1]
    p4 = rectangle[1].copy()
    res = (function(*p1), function(*p2), function(*p3), function(*p4))
    if return_points:
        return res, (p1, p2, p3, p4)
    else:
        return res


def load_joblib(path: Union[str, Path]):
    with open(str(path), "r") as f:
        return joblib.load(f.name)


def save_joblib(path: Union[str, Path], variable):
    with open(str(path), "w") as f:
        joblib.dump(variable, f.name)


class DatasetsBaseManager:
    def __init__(self, path2data: Union[str, Path], curve: Type[CurveBase], N: int, kernel_size: Tuple[int, int],
                 min_val: float, max_val: float, velocity_range: Tuple[Tuple, Tuple], recalculate=False, workers=-1,
                 seed=42, reload_data=True):
        self.path2datafolder = Path(path2data)
        self.path2datafolder.mkdir(parents=True, exist_ok=True)
        self.workers = workers

        self.curve = curve
        self.N = N
        self.kernel_size = kernel_size

        self.recalculate = recalculate
        self.seed = seed

        # TODO: generalize method to receive a PDE instead of a velocity.
        assert np.all(np.diff(velocity_range, axis=0) >= 0), "max vel > min_vel should be."
        assert np.all(np.array(velocity_range) >= 0), "vel >= 0 should be."
        # assert np.all(np.array(velocity_range)[:, 0] > 0), "vel.x > 0 should be."
        assert np.all(np.array(velocity_range) <= 1), "max vel < 1 should be."
        self.velocity_range = velocity_range  # ((vx_min, vy_min), (vx_max, vy_max)) in fractions of discretisation
        self.center_cell_coords = np.array(kernel_size) // 2
        self.center = self.get_center(kernel_size)

        self.min_val = min_val
        self.max_val = max_val

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

    # --------- file properties ---------- #
    @property
    def base_name(self):
        kernel_name = "_".join(map(str, self.kernel_size))
        return f"{self.curve.__name__}_k{kernel_name}_n{self.N}_min{self.min_val}_max{self.max_val}_" \
               f"v{clean_str4saving(str(self.velocity_range))}"

    @property
    def name4learning(self):
        """In case special data aboit the transformations has to be added."""
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
        assert n <= self.N, f"Number of examples to use should be less then {self.N}"
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
        return {k: v[:n] if isinstance(v, List) else v for k, v in data.items()}

    def transform_curve_data(self, *args):
        """
        The data may be saved always in the same format but transformed to get different Learning objectives.
        """
        return args

    def get_dataset4problem(self, type_of_problem, n=None, training_noise=0, n_train=None):
        n = self.N if n is None else n
        data = self.load_dataset(n)
        noise = np.random.normal(loc=0, scale=training_noise,
                                 size=tuple([len(data["kernel"])] + list(self.kernel_size)))
        if type_of_problem == CLASSIFICATION_PROBLEM:
            input_data = data["kernel"] + noise
            output_data = data["classification"]
        elif type_of_problem == FLUX_PROBLEM:
            input_data = list(zip(data["kernel"] + noise, data["velocity"]))
            output_data = data["flux"]
        elif type_of_problem == CURVE_PROBLEM:
            input_data = data["kernel"] + noise
            output_data = np.transpose(self.transform_curve_data(*np.transpose(data["curve"])))
        else:
            raise Exception("Type of problem {} not implemented.".format(type_of_problem))
        n_train = n if n_train is None else n_train
        return input_data[:n_train], output_data[:n_train], input_data[n_train:], output_data[n_train:]

    # --------- save ---------- #
    def save_dataset(self, data):
        save_joblib(self.path2data, data)

    # --------- generate data ---------- #
    def get_velocity(self):
        return np.random.uniform(*self.velocity_range)

    def get_curve_data(self):
        """
        The center of the stencil is in the center of the kernel_size: //2+0.5
        -> go to DatasetsbaseManager generate_dataset
        """
        raise Exception("Not implemented.")

    def get_curve(self, curve_data):
        return self.curve(*curve_data)

    def generate_dataset(self):
        print(f"Generating data {self.base_name}...")
        np.random.seed(self.seed)

        def par_func(params):
            curve_data, velocity = params
            curve = self.get_curve(curve_data)

            # Calculate averages
            # The coordinates must be so the origin is in the center of the central cell
            kernel = np.zeros(self.kernel_size)
            for coords in map(np.array, itertools.product(*list(map(range, self.kernel_size)))):
                centered_coords = coords - self.center_cell_coords - 0.5
                kernel[tuple(coords)] = curve.calculate_rectangle_average(
                    x_limits=(centered_coords[0], centered_coords[0] + 1.0),
                    y_limits=(centered_coords[1], centered_coords[1] + 1.0)
                )

            # is it a Regular cell or a curve cell?
            # Assumes for non functions curves that functions to train are 0 to 1.
            classification = [(0 < kernel[self.kernel_size[0] // 2, self.kernel_size[1] // 2]) and
                              (1 > kernel[self.kernel_size[0] // 2, self.kernel_size[1] // 2])]

            # calculate flux
            next_coords, next_rectangles = get_rectangles_and_coords_to_calculate_flux(
                coords=np.array(self.kernel_size) // 2,
                velocity=velocity
            )
            flux = [curve.calculate_rectangle_average(
                x_limits=(rectangle[0][0], rectangle[1][0]),
                y_limits=(rectangle[0][1], rectangle[1][1])
            ) for rectangle in next_rectangles]
            # curve_data[:-2] because the last two are the values of each side
            # TODO: harcoded only one direction
            return kernel, velocity, classification, np.ravel(curve_data[:-2]), [flux[1]]

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
                                 value_down) -> CurveBase:
        raise Exception("Not implemented.")
