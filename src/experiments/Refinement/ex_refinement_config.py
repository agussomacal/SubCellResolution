from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

import config
from experiments.global_params import cred

experiment_path = config.results_path.joinpath("Subdivision")
experiment_path.mkdir(parents=True, exist_ok=True)

plt.style.use(Path(__file__).parent.joinpath("refinement.mplstyle"))



# --------- Colors and models --------- #
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_BROWN, C_PINK, C_GRAY, C_OLIVE, C_CYAN = (
    matplotlib.colormaps['tab10'].colors)
C_BLACK = (0, 0, 0)
C_WHITE = (1, 1, 1)
