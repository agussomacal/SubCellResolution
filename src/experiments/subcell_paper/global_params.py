from PerplexityLab.LaTexReports import RunsInfo2Latex
from PerplexityLab.miscellaneous import ClassPartialInit
from config import subcell_paper_folder_path
from lib.Curves.AverageCurves import CurveAveragePolynomial

from seaborn import color_palette

cblue, corange, cgreen, cred, cpurple, cbrown, cpink, cgray, cyellow, ccyan = color_palette("tab10")

SUB_CELL_DISCRETIZATION2BOUND_ERROR = 10
OBERA_ITERS = 500
CCExtraWeight = 100  # central cell extra weight 100

CurveAverageQuadraticCC = ClassPartialInit(CurveAveragePolynomial, class_name="CurveAverageQuadraticCC",
                                           degree=2, ccew=CCExtraWeight)

runsinfo = RunsInfo2Latex(path2latex=f"{subcell_paper_folder_path}/main.tex")
runsinfo.insert_preamble_in_latex_file()
runsinfo.append_info(
    cceweight=CCExtraWeight
)
