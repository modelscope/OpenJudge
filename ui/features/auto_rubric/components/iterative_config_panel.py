# -*- coding: utf-8 -*-
"""Iterative Rubric configuration panel for Auto Rubric feature.

Provides UI for configuring Iterative Rubric generation:
- Grader name
- Data upload
- Task description (optional)
- Advanced settings (categorization, etc.)
"""

from typing import Any

import streamlit as st
from features.auto_rubric.components.data_upload_panel import render_data_upload_panel
from shared.i18n import t


def render_iterative_config_panel(sidebar_config: dict[str, Any]) -> dict[str, Any]:
    """Render the Iterative Rubric configuration panel.

    Args:
        sidebar_config: Configuration from the sidebar.

    Returns:
        Complete configuration dictionary including:
        - grader_name: Name for the generated grader
        - dataset: Parsed training data
        - task_description: Optional task description
        - enable_categorization: Whether to group rubrics
        - categories_number: Number of categories
        - All sidebar config values
    """
    config: dict[str, Any] = {}
    config.update(sidebar_config)

    # Grader Name
    grader_name = st.text_input(
        t("rubric.config.grader_name"),
        placeholder=t("rubric.config.grader_name_placeholder"),
        help=t("rubric.config.grader_name_help"),
        key="rubric_iterative_grader_name",
    )

    # Data Upload Section
    st.markdown('<div style="margin: 1rem 0;"></div>', unsafe_allow_html=True)

    upload_result = render_data_upload_panel(mode=sidebar_config.get("grader_mode", "pointwise"))

    # Task Description (optional for iterative mode)
    st.markdown('<div style="margin: 1rem 0;"></div>', unsafe_allow_html=True)

    with st.expander(f"📝 {t('rubric.iterative.task_desc_title')}", expanded=False):
        task_description = st.text_area(
            t("rubric.config.task_description"),
            placeholder=t("rubric.iterative.task_desc_placeholder"),
            height=100,
            help=t("rubric.iterative.task_desc_help"),
            key="rubric_iterative_task_desc",
        )

    # Advanced Settings
    with st.expander(f"⚙️ {t('rubric.iterative.advanced')}", expanded=False):
        st.markdown(
            f"<div style='color: #94A3B8; font-size: 0.85rem; margin-bottom: 0.5rem;'>"
            f"{t('rubric.iterative.advanced_desc')}</div>",
            unsafe_allow_html=True,
        )

        enable_categorization = st.checkbox(
            t("rubric.iterative.enable_categorization"),
            value=True,
            help=t("rubric.iterative.enable_categorization_help"),
            key="rubric_enable_categorization",
        )

        if enable_categorization:
            categories_number = st.slider(
                t("rubric.iterative.categories_number"),
                min_value=2,
                max_value=10,
                value=5,
                help=t("rubric.iterative.categories_number_help"),
                key="rubric_categories_number",
            )
        else:
            categories_number = 5

        query_specific_number = st.slider(
            t("rubric.iterative.query_specific_number"),
            min_value=1,
            max_value=5,
            value=2,
            help=t("rubric.iterative.query_specific_number_help"),
            key="rubric_query_specific_number",
        )

    config["grader_name"] = grader_name
    config["dataset"] = upload_result.get("data")
    config["data_count"] = upload_result.get("count", 0)
    config["data_valid"] = upload_result.get("is_valid", False)
    config["task_description"] = task_description if task_description else None
    config["enable_categorization"] = enable_categorization
    config["categories_number"] = categories_number
    config["query_specific_generate_number"] = query_specific_number

    return config


def validate_iterative_config(config: dict[str, Any]) -> tuple[bool, str]:
    """Validate Iterative Rubric configuration.

    Args:
        config: Configuration dictionary to validate.

    Returns:
        Tuple of (is_valid, error_message).
    """
    if not config.get("grader_name", "").strip():
        return False, t("rubric.validation.name_required")

    if not config.get("data_valid", False):
        return False, t("rubric.validation.data_required")

    if not config.get("api_key", "").strip():
        return False, t("rubric.validation.api_key_required")

    if not config.get("model_name", "").strip():
        return False, t("rubric.validation.model_required")

    # Check minimum data count
    data_count = config.get("data_count", 0)
    if data_count < 10:
        return False, t("rubric.validation.min_data_required", count=10)

    return True, ""
