# -*- coding: utf-8 -*-
"""Sidebar component for Auto Rubric feature.

Provides configuration for:
- LLM API settings (provider, key, model)
- Generation settings (language, evaluation mode, score range)
- Advanced settings (max retries)
"""

from typing import Any

import streamlit as st
from shared.constants import DEFAULT_API_ENDPOINTS, DEFAULT_MODELS
from shared.i18n import t


def _render_llm_config(config: dict[str, Any]) -> None:
    """Render LLM configuration section."""
    st.markdown(
        f'<div class="section-header">{t("rubric.sidebar.llm_config")}</div>',
        unsafe_allow_html=True,
    )

    # API Provider selection
    provider = st.selectbox(
        t("api.provider"),
        options=list(DEFAULT_API_ENDPOINTS.keys()),
        index=0,
        help=t("rubric.sidebar.provider_help"),
        key="rubric_api_provider",
    )

    # Custom endpoint input (only for Custom provider)
    if provider == "Custom":
        api_endpoint = st.text_input(
            t("api.custom_endpoint"),
            placeholder=t("api.custom_endpoint_placeholder"),
            help=t("api.custom_endpoint_help"),
            key="rubric_custom_endpoint",
        )
    else:
        api_endpoint = DEFAULT_API_ENDPOINTS[provider]

    # API Key input
    api_key = st.text_input(
        t("api.key"),
        type="password",
        placeholder=t("api.key_placeholder"),
        help=t("rubric.sidebar.api_key_help"),
        key="rubric_api_key",
    )

    if api_key:
        st.success(t("api.key_configured"))
    else:
        st.warning(t("api.key_required"))

    # Model selection
    model_option = st.selectbox(
        t("model.select"),
        options=DEFAULT_MODELS + [t("model.custom")],
        index=0,
        help=t("rubric.sidebar.model_help"),
        key="rubric_model",
    )

    if model_option == t("model.custom"):
        model_name = st.text_input(
            t("model.custom_input"),
            placeholder=t("model.custom_placeholder"),
            key="rubric_custom_model",
        )
    else:
        model_name = model_option

    config["api_endpoint"] = api_endpoint
    config["api_key"] = api_key
    config["model_name"] = model_name


def _render_generation_settings(config: dict[str, Any]) -> None:
    """Render generation settings section."""
    st.markdown(
        f'<div class="section-header">{t("rubric.sidebar.gen_settings")}</div>',
        unsafe_allow_html=True,
    )

    # Language selection
    language_options = {"English": "EN", "中文": "ZH"}
    language_display = st.selectbox(
        t("rubric.sidebar.language"),
        options=list(language_options.keys()),
        index=0,
        help=t("rubric.sidebar.language_help"),
        key="rubric_language",
    )
    language = language_options[language_display]

    # Evaluation mode
    mode_options = {
        t("rubric.sidebar.pointwise"): "pointwise",
        t("rubric.sidebar.listwise"): "listwise",
    }
    mode_display = st.selectbox(
        t("rubric.sidebar.eval_mode"),
        options=list(mode_options.keys()),
        index=0,
        help=t("rubric.sidebar.eval_mode_help"),
        key="rubric_eval_mode",
    )
    grader_mode = mode_options[mode_display]

    # Score range (only for pointwise mode)
    if grader_mode == "pointwise":
        st.markdown(
            f'<div style="font-size: 0.85rem; color: #94A3B8; margin-bottom: 0.25rem;">'
            f'{t("rubric.sidebar.score_range")}</div>',
            unsafe_allow_html=True,
        )
        col1, col2 = st.columns(2)
        with col1:
            min_score = st.number_input(
                t("rubric.sidebar.min_score"),
                min_value=0,
                max_value=100,
                value=0,
                step=1,
                key="rubric_min_score",
            )
        with col2:
            max_score = st.number_input(
                t("rubric.sidebar.max_score"),
                min_value=1,
                max_value=100,
                value=5,
                step=1,
                key="rubric_max_score",
            )
    else:
        min_score = 0
        max_score = 1  # Not used for listwise

    config["language"] = language
    config["grader_mode"] = grader_mode
    config["min_score"] = min_score
    config["max_score"] = max_score


def _render_advanced_settings(config: dict[str, Any]) -> None:
    """Render advanced settings section."""
    with st.expander(t("rubric.sidebar.advanced"), expanded=False):
        max_retries = st.number_input(
            t("rubric.sidebar.max_retries"),
            min_value=1,
            max_value=10,
            value=3,
            step=1,
            help=t("rubric.sidebar.max_retries_help"),
            key="rubric_max_retries",
        )
        config["max_retries"] = max_retries


def render_rubric_sidebar() -> dict[str, Any]:
    """Render the Auto Rubric sidebar and return configuration.

    Returns:
        Dictionary containing all sidebar configuration values:
        - api_endpoint: API endpoint URL
        - api_key: API key for authentication
        - model_name: Model name to use
        - language: Language for generation (EN/ZH)
        - grader_mode: Evaluation mode (pointwise/listwise)
        - min_score: Minimum score (pointwise only)
        - max_score: Maximum score (pointwise only)
        - max_retries: Maximum retry attempts
    """
    config: dict[str, Any] = {}

    _render_llm_config(config)
    _render_generation_settings(config)
    _render_advanced_settings(config)

    return config
