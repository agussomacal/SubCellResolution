from pathlib import Path

import matplotlib
from matplotlib import pyplot as plt

import config

experiment_name = "Subdivision"
experiment_subdivision_path = config.results_path.joinpath("Subdivision")
experiment_subdivision_path.mkdir(parents=True, exist_ok=True)

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

# --------- Plot parameters --------- #
axis_font_dict = {'color': 'black', 'weight': 'normal', 'size': 25}
labels_font_dict = {'color': 'black', 'weight': 'normal', 'size': 25}
legend_font_dict = {'weight': 'normal', "size": 19, 'stretch': 'normal'}
line_style = {1: "solid", 2: "dashed", 3: "dashdot", 4: "dotted"}
marker_style = {1: "o", 2: "^", 3: "s", 4: ""}

# --------- Names and colors --------- #
experiment_labels_order = ["ELVIRA", "ELVIRA W", "AEROS linear", "AEROS quadratic", "AEROS cubic", "AEROS quartic"]

color = {
    "ELVIRA": C_ORANGE,
    "ELVIRA W": C_RED,
    "AEROS linear": C_BLUE,
    "AEROS quadratic": C_GREEN,
    "AEROS cubic": C_PURPLE,
    "AEROS quartic": C_PINK,
}

method_name = {
    "ELVIRA": "ELVIRA",
    "ELVIRA W": "ELVIRA W",
    "AEROS linear": "AEROS linear",
    "AEROS quadratic": "AEROS quadratic",
    "AEROS cubic": "AEROS cubic",
    "AEROS quartic": "AEROS quartic",
}

# --------- Reconstruction plot params --------- #
curve_color = C_RED
cmap_reconstruction = "Reds"
cmap_true_image = "Greys_r"
fig_size = (15, 15)
