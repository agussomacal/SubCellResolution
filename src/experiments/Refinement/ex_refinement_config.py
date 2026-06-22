from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

import config
from experiments.global_params import cred

experiment_name = "Subdivision"
experiment_path = config.results_path.joinpath("Subdivision")
experiment_path.mkdir(parents=True, exist_ok=True)

# --------- Matplotlib configs --------- #
plt.style.use(Path(__file__).parent.joinpath("refinement.mplstyle"))
# For latex compilation problem: https://search.brave.com/search?q=cm-super&summary=1&conversation=093c8ce05a8ce05ba3ffa1e62041e5146bab
# apt install cm-super
packages = ("amsmath",)
plt.rc(
    'text.latex',
    preamble=r''.join([f"\\usepackage{{{package}}}" for package in packages])
)

# --------- Colors and models --------- #
C_BLUE, C_ORANGE, C_GREEN, C_RED, C_PURPLE, C_BROWN, C_PINK, C_GRAY, C_OLIVE, C_CYAN = (
    matplotlib.colormaps['tab10'].colors)
C_BLACK = (0, 0, 0)
C_WHITE = (1, 1, 1)
