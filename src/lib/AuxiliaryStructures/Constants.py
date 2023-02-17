import numpy as np
import seaborn as sns

# --------- BaseEdgeDetector Constants --------- #
REGULAR_CELL = 0
CURVE_CELL = 1
VERTEX_CELL = -1  # 2
TRIPLE_POINT_CELL = -1  # 3

NUM2CELL_TYPE = {
    REGULAR_CELL: "Regular cell",
    CURVE_CELL: "Curve cell",
    VERTEX_CELL: "Vertex cell",
    TRIPLE_POINT_CELL: "Truple point cell",
}

VERTICAL = 1
HORIZONTAL = 0

# clockwise order first touching edge, then touching point
NEIGHBOURHOOD_8_MANHATTAN = np.array([[1, 0], [0, 1], [-1, 0], [0, -1],
                                      [1, 1], [1, -1], [-1, -1], [-1, 1]])
NEIGHBOURHOOD_8 = np.array([[1, 1], [1, 0], [1, -1], [0, -1],
                            [-1, -1], [-1, 0], [-1, 1], [0, 1]])

neighbourhood_8_dict2ix = {8: 0, 7: 1, 6: 2, 3: 3, 0: 4, 1: 5, 2: 6, 5: 7}


def neighbourhood_8_ix(vec: np.ndarray):
    return neighbourhood_8_dict2ix[int(((1 + vec) * np.array([3, 1])).sum())]


# --------- Colors --------- #
# greenish
OLD_OLIVE = (138 / 255, 151 / 255, 71 / 255)
NIGHT_NAVY = (33 / 255, 64 / 255, 95 / 255)
ISLAND_INDIGO = (0, 126 / 255, 135 / 255)

# bluish
PACIFIC_POINT = (0, 126 / 255, 135 / 255)

# redish
REAL_RED = (198 / 255, 44 / 255, 58 / 255)

# violetish
ELEGANT_EGGPLANT = (96 / 255, 74 / 255, 110 / 255)

cmap = sns.color_palette("colorblind")
CBLIND_BLUE = sns.color_palette("colorblind")[0]
CBLIND_ORANGE = sns.color_palette("colorblind")[1]
CBLIND_GREEN = sns.color_palette("colorblind")[2]
CBLIND_RED = sns.color_palette("colorblind")[3]
CBLIND_VIOLET = sns.color_palette("colorblind")[4]
CBLIND_BROWN = sns.color_palette("colorblind")[5]
CBLIND_PINK = sns.color_palette("colorblind")[6]
CBLIND_GRAY = sns.color_palette("colorblind")[7]
CBLIND_YELLOW = sns.color_palette("colorblind")[8]
CBLIND_CYAN = sns.color_palette("colorblind")[9]
