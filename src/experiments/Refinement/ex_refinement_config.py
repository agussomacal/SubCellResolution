import matplotlib

import config
from experiments.global_params import cred

experiment_path = config.results_path.joinpath("Subdivision")
experiment_path.mkdir(parents=True, exist_ok=True)

# Reconstruction plot params
matplotlib.rcParams['text.usetex'] = False
curve_color = cred
cmap_reconstruction = "Reds"
cmap_true_image = "Greys_r"
fig_size = (15, 15)
