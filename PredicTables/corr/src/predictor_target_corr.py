from .ctsX_ctsY import calc_continuousX_continuousY_corr
from .ctsX_binY import calc_continuousX_binaryY_corr
from .ctsX_catY import calc_continuousX_categoricalY_corr
from .binX_binY import calc_binaryX_binaryY_corr
from .binX_catY import calc_binaryX_categoricalY_corr
from .catX_catY import calc_categoricalX_categoricalY_corr


def predictor_target_corr(dataset, target_variable):
    return calc_predictor_target_corr(dataset, target_variable)
