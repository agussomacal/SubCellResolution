import time

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pylab as plt

import config
from experiments.models import piecewise_constant, calculate_averages_from_image, load_image
from lib.SubCellReconstruction import SubCellReconstruction
from src.DataManager import DataManager, JOBLIB
from src.LabPipeline import LabPipeline
from src.viz_utils import perplex_plot

SUB_DISCRETIZATION2BOUND_ERROR = 5


def calculate_reconstruction_error(image: np.ndarray, model: SubCellReconstruction):
    image = load_image(image)
    im_shape = np.array(np.shape(image))
    # find the maximum divisor for each dim shape near SUB_DISCRETIZATION2BOUND_ERROR
    model_resolution = np.array(model.resolution)
    resolution_factor = np.array(
        (im_shape / model_resolution) // ((im_shape / model_resolution) / SUB_DISCRETIZATION2BOUND_ERROR),
        dtype=int)
    assert np.min(resolution_factor) >= SUB_DISCRETIZATION2BOUND_ERROR - 1, "Sub discretization not fine enough"
    # assert np.min(
    #     resolution_factor4model) >= SUB_DISCRETIZATION2BOUND_ERROR - 1, "Sub discretization not fine enough"
    # TODO: in case of full experiment, option to full discretization.
    true_reconstruction = calculate_averages_from_image(image, model_resolution * resolution_factor)
    t0 = time.time()
    reconstruction = model.reconstruct(resolution_factor=resolution_factor)
    dt = time.time() - t0
    reconstruction_error = np.abs(np.array(reconstruction) - true_reconstruction)
    return {
        "reconstruction_error": reconstruction_error,
        "time_to_sr_reconstruction": dt
    }


@perplex_plot
def plot_convergence_curves(fig, ax, num_cells_per_dim, reconstruction_error, fit_models):
    data = pd.DataFrame.from_dict({
        "N": np.array(num_cells_per_dim) ** 2,
        "Error": list(map(np.mean, reconstruction_error)),
        "Model": fit_models
    })
    for model, d in data.groupby("Model"):
        plt.plot(d["N"], d["Error"], label=model)
    ax.set_yscale("log")
    ax.set_xscale("log")
    ax.legend()


if __name__ == "__main__":
    data_manager = DataManager(
        path=config.results_path,
        name='ImageReconstruction',
        format=JOBLIB
    )

    lab = LabPipeline()
    lab.define_new_block_of_functions(
        "fit_models",
        piecewise_constant
    )
    lab.define_new_block_of_functions(
        "ReconstructModels",
        calculate_reconstruction_error
    )

    lab.execute(
        data_manager,
        num_cores=1,
        recalculate=False,
        forget=False,
        refinement=[1],
        num_cells_per_dim=[21, 28, 42],
        noise=[0],
        image=["ShapesVertex_1680x1680.jpg"]
    )

    plot_convergence_curves(data_manager)
