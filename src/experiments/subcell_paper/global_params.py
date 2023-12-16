import psutil
from seaborn import color_palette

from PerplexityLab.LaTexReports import RunsInfo2Latex
from PerplexityLab.miscellaneous import ClassPartialInit
from config import subcell_paper_folder_path
from lib.Curves.AverageCurves import CurveAveragePolynomial
from lib.Curves.VanderCurves import CurveVandermondePolynomial

cblue, corange, cgreen, cred, cpurple, cbrown, cpink, cgray, cyellow, ccyan = color_palette("tab10")

bluish = (17, 110, 138)
redish = (153, 0, 91)
pinkish = (230, 0, 136)
greenish = (6, 194, 88)
cyanish = (82, 189, 236)

EVALUATIONS = True

SUB_CELL_DISCRETIZATION2BOUND_ERROR = 20
OBERA_ITERS = 500
CCExtraWeight = 100  # central cell extra weight 100

CurveAverageQuadraticCC = ClassPartialInit(CurveAveragePolynomial, class_name="CurveAverageQuadraticCC",
                                           degree=2, ccew=CCExtraWeight)
VanderQuadratic = ClassPartialInit(CurveVandermondePolynomial, class_name="VanderQuadratic", degree=2)
AvgQuadratic = ClassPartialInit(CurveAveragePolynomial, class_name="AvgQuadratic", degree=2)

runsinfo = RunsInfo2Latex(path2latex=f"{subcell_paper_folder_path}/main.tex")
runsinfo.insert_preamble_in_latex_file()
runsinfo.append_info(
    cceweight=CCExtraWeight
)

RAM = psutil.virtual_memory().total / 1000000000  # in GB
if RAM > 70:  # if run in server
    running_in = "server"
    print("running in server")
    N = int(1e6)
    num_cores = workers = 50
    RESOLUTION_FACTOR = 1
elif RAM > 40:
    running_in = "local-power"
    print("running in local power machine")
    N = int(1e6)
    num_cores = workers = 15
    RESOLUTION_FACTOR = 5
else:
    running_in = "local"
    print("running in local machine")
    N = int(1e4)
    num_cores = workers = 3
    RESOLUTION_FACTOR = 5
