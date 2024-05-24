"""Show fit statistics and plots for the model."""

import streamlit as st

from predictables.app import fit_model_with_cross_validation
from predictables.app.src.util import get_data
from predictables.app.plots.roc import roc_curve_general


models = fit_model_with_cross_validation(True, get_data())
target_variable = st.session_state["target_variable"]

st.markdown("# Model Evaluation")

