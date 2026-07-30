"""Metrics page: the dataset fleet first, then the active run's pipeline detail.

The inventory answers "what do I have and which set is cleanest" across every
import; the panel below it answers "how did this one dataset get here" — durations,
throughput, errors, and GPU cost per step.

This page can be viewed anytime and is updated as each step completes.
"""

import streamlit as st
from components import inventory_panel, observability_panel

inventory_panel.render()
st.divider()
observability_panel.render()
