from PerplexityLab.miscellaneous import ClassPartialInit
from lib.Curves.AverageCurves import CurveAveragePolynomial

SUB_CELL_DISCRETIZATION2BOUND_ERROR = 10
OBERA_ITERS = 500
CCExtraWeight = 100  # central cell extra weight

CurveAverageQuadraticCC = ClassPartialInit(CurveAveragePolynomial, class_name="CurveAverageQuadraticCC",
                                           degree=2, ccew=CCExtraWeight)
