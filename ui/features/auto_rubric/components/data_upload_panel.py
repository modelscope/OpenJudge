# -*- coding: utf-8 -*-
"""Data upload panel for Iterative Rubric mode.

Provides UI for uploading and previewing labeled training data.
"""

import html
from typing import Any

import streamlit as st
from features.auto_rubric.services.data_parser import DataParser
from shared.i18n import t


def _escape_html(text: str) -> str:
    """Escape HTML special characters."""
    return html.escape(str(text)) if text else ""


def render_data_upload_panel(mode: str = "pointwise") -> dict[str, Any]:
    """Render the data upload panel for Iterative Rubric mode.

    Args:
        mode: Evaluation mode ("pointwise" or "listwise").

    Returns:
        Dictionary containing:
        - is_valid: Whether valid data has been uploaded
        - data: Parsed data list (if valid)
        - count: Number of records
    """
    result: dict[str, Any] = {
        "is_valid": False,
        "data": None,
        "count": 0,
    }

    st.markdown(
        f"""
        <div style="
            font-weight: 600;
            color: #F1F5F9;
            margin-bottom: 0.5rem;
        ">{t('rubric.upload.title')}</div>
        """,
        unsafe_allow_html=True,
    )

    # Format requirements info
    if mode == "pointwise":
        required_fields = "query, response, label_score"
    else:
        required_fields = "query, responses, label_rank"

    st.markdown(
        f"""
        <div style="
            background: rgba(30, 41, 59, 0.5);
            border: 1px solid #334155;
            border-radius: 8px;
            padding: 0.75rem;
            margin-bottom: 0.75rem;
            font-size: 0.85rem;
        ">
            <div style="color: #94A3B8; margin-bottom: 0.25rem;">
                {t('rubric.upload.format_hint')}
            </div>
            <div style="color: #A5B4FC;">
                {t('rubric.upload.required_fields')}: <code>{required_fields}</code>
            </div>
            <div style="color: #64748B; font-size: 0.8rem; margin-top: 0.25rem;">
                {t('rubric.upload.supported_formats')}
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # File uploader
    uploaded_file = st.file_uploader(
        t("rubric.upload.label"),
        type=["json", "jsonl", "csv"],
        help=t("rubric.upload.help"),
        key="rubric_data_upload",
    )

    if uploaded_file is not None:
        # Parse the file
        parser = DataParser()
        content = uploaded_file.read()

        with st.spinner(t("rubric.upload.parsing")):
            parse_result = parser.parse_file(
                file_content=content,
                filename=uploaded_file.name,
                mode=mode,
            )

        if parse_result.success and parse_result.data:
            # Show success message
            st.success(t("rubric.upload.success", count=parse_result.total_count))

            result["is_valid"] = True
            result["data"] = parse_result.data
            result["count"] = parse_result.total_count

            # Show warnings if any
            if parse_result.warnings:
                with st.expander(
                    f"⚠️ {t('rubric.upload.warnings')} ({len(parse_result.warnings)})",
                    expanded=False,
                ):
                    for warning in parse_result.warnings[:10]:
                        st.caption(f"• {_escape_html(warning)}")
                    if len(parse_result.warnings) > 10:
                        st.caption(f"... {len(parse_result.warnings) - 10} more")

            # Show data preview
            with st.expander(
                f"👁️ {t('rubric.upload.preview')} ({min(3, parse_result.total_count)} / {parse_result.total_count})",
                expanded=True,
            ):
                preview = parser.get_preview(parse_result.data, max_items=3)
                for i, item in enumerate(preview, 1):
                    st.markdown(
                        f"<div style='color: #A5B4FC; font-weight: 500; margin-top: 0.5rem;'>" f"Record {i}</div>",
                        unsafe_allow_html=True,
                    )
                    # Display each field
                    for key, value in item.items():
                        display_value = _escape_html(str(value))
                        if len(display_value) > 150:
                            display_value = display_value[:150] + "..."
                        st.markdown(
                            f"<div style='font-size: 0.85rem;'>"
                            f"<span style='color: #94A3B8;'>{_escape_html(key)}:</span> "
                            f"<span style='color: #E2E8F0;'>{display_value}</span></div>",
                            unsafe_allow_html=True,
                        )
        else:
            # Show error
            error_msg = parse_result.error or "Unknown error"
            st.error(f"{t('rubric.upload.error')}: {_escape_html(error_msg)}")

            if parse_result.warnings:
                with st.expander(t("rubric.upload.details"), expanded=False):
                    for warning in parse_result.warnings[:10]:
                        st.caption(f"• {_escape_html(warning)}")

    else:
        # Show placeholder
        st.markdown(
            f"""
            <div style="
                background: rgba(30, 41, 59, 0.3);
                border: 2px dashed #475569;
                border-radius: 8px;
                padding: 2rem;
                text-align: center;
                margin-top: 0.5rem;
            ">
                <div style="font-size: 1.5rem; margin-bottom: 0.5rem;">📂</div>
                <div style="color: #94A3B8; font-size: 0.9rem;">
                    {t('rubric.upload.drop_hint')}
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return result
