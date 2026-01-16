# -*- coding: utf-8 -*-
"""UI Components module for OpenJudge Studio."""

from components.input_panel import render_input_panel
from components.result_panel import render_result_panel
from components.shared import render_footer, render_header, render_quick_guide
from components.sidebar import render_sidebar

__all__ = [
    "render_sidebar",
    "render_input_panel",
    "render_result_panel",
    "render_header",
    "render_footer",
    "render_quick_guide",
]
