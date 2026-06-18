import time
from dataclasses import dataclass
from typing import Callable

import numpy as np
from matplotlib import pyplot as plt

from experiments.Refinement.ex_refinement_config import experiment_path
from experiments.global_params import EVALUATIONS
from experiments.tools import calculate_averages_from_image, load_image, reconstruct
from lib.AuxiliaryStructures.Indexers import ArrayIndexerNd
from perplexitylab.miscellaneous import check_do_save_or_load_experiment


@dataclass
class ExperimentInfo:
    label: str
    image_name: str
    num_cells_per_dim: int
    sub_cell_model: Callable
    refinement: int = 1
    reconstruction_factor: int = 1
    angle_threshold: float = 0.
    # color: Union[str, Tuple] = "black"
    # linestyle: str = 'dashed'
    # marker: str = '.'
    recalculate: bool = False

    @property
    def identifier(self):
        return f"Img{self.image_name.split('.')[0]}_{self.num_cells_per_dim}x{self.num_cells_per_dim}_{self.label}_Ref{self.refinement}"


def single_experiment_vars_filter(experiment_info: ExperimentInfo, *args, **kwargs):
    return (experiment_info.identifier,)


def image_to_avg(experiment_info: ExperimentInfo, image, noise=0, seed=42):
    avg_values = calculate_averages_from_image(image, experiment_info.num_cells_per_dim)
    np.random.seed(seed)
    return avg_values + np.random.uniform(-noise, noise, size=avg_values.shape)


@check_do_save_or_load_experiment(default_path=experiment_path, vars_filter=single_experiment_vars_filter)
def fit_model(experiment_info: ExperimentInfo, avg_values):
    model = experiment_info.sub_cell_model(refinement=experiment_info.refinement,
                                           angle_threshold=experiment_info.angle_threshold)

    t0 = time.time()
    model.fit(average_values=avg_values, indexer=ArrayIndexerNd(avg_values, "cyclic"))
    t_fit = time.time() - t0
    print("Time to fit model:", t_fit)
    return model


@check_do_save_or_load_experiment(default_path=experiment_path, vars_filter=single_experiment_vars_filter,
                                  saver=lambda data, filepath: plt.imsave(filepath, data),
                                  loader=lambda filepath: load_image(filepath, other_path=""),
                                  file_format="png")
def do_reconstruction(experiment_info: ExperimentInfo, image, model):
    t0 = time.time()
    reconstruction = reconstruct(image, model.cells, model.resolution, experiment_info.reconstruction_factor,
                                 do_evaluations=EVALUATIONS)
    t_reconstruct = time.time() - t0
    print("Time to reconstruct:", t_reconstruct)
    return reconstruction
