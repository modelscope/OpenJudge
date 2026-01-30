# -*- coding: utf-8 -*-
"""Workspace selector component for multi-user isolation."""

import streamlit as st
from shared.i18n import t
from shared.services.workspace_manager import (
    STATE_CURRENT_WORKSPACE,
    WorkspaceManager,
    get_current_workspace,
    set_current_workspace,
)

# Session state keys for workspace UI
STATE_SHOW_CREATE_DIALOG = "workspace_show_create_dialog"
STATE_SHOW_DELETE_CONFIRM = "workspace_show_delete_confirm"


def render_workspace_selector() -> None:
    """Render the workspace selector in the sidebar.

    Allows users to:
    - See current workspace
    - Switch to a different workspace
    - Create new named workspaces
    """
    manager = WorkspaceManager.get_instance()
    current_ws = get_current_workspace()

    # Determine display name for current workspace
    if current_ws.startswith(manager.ANONYMOUS_PREFIX):
        current_display = t("workspace.anonymous")
    else:
        current_display = current_ws

    # Main selector row
    col_label, col_ws, col_menu = st.columns([1.2, 3.5, 1.3])

    with col_label:
        st.markdown(
            f'<div style="font-size: 0.75rem; font-weight: 600; color: #64748B; '
            f'padding-top: 0.5rem;">{t("workspace.label")}</div>',
            unsafe_allow_html=True,
        )

    with col_ws:
        # List available workspaces
        workspaces = manager.list_workspaces(include_anonymous=False)
        ws_names = [ws["name"] for ws in workspaces]

        # Build options: current anonymous + named workspaces
        options = []
        option_display = {}

        # Add anonymous option (current user's default)
        anon_key = "__anonymous__"
        options.append(anon_key)
        option_display[anon_key] = f"🔒 {t('workspace.anonymous')}"

        # Add named workspaces
        for ws in workspaces:
            options.append(ws["name"])
            option_display[ws["name"]] = f"📁 {ws['display_name']}"

        # Determine current index
        if current_ws.startswith(manager.ANONYMOUS_PREFIX):
            current_index = 0
        elif current_ws in ws_names:
            current_index = ws_names.index(current_ws) + 1
        else:
            current_index = 0

        selected = st.selectbox(
            t("workspace.select"),
            options=options,
            index=current_index,
            format_func=lambda x: option_display.get(x, x),
            key="workspace_selector",
            label_visibility="collapsed",
        )

        # Handle workspace switch
        if selected == anon_key:
            browser_id = st.session_state.get("workspace_browser_id", "default")
            target_ws = manager.get_anonymous_workspace_name(browser_id)
        else:
            target_ws = selected

        if target_ws != current_ws:
            set_current_workspace(target_ws)
            manager.ensure_workspace_exists(target_ws)
            st.rerun()

    with col_menu:
        with st.popover("⚙️", use_container_width=True):
            st.markdown(f"**{t('workspace.manage')}**")

            # Create new workspace
            if st.button(f"➕ {t('workspace.create')}", key="ws_create_btn", use_container_width=True):
                st.session_state[STATE_SHOW_CREATE_DIALOG] = True

            # Delete workspace (only if named workspace is selected)
            is_named_ws = not current_ws.startswith(manager.ANONYMOUS_PREFIX)
            if st.button(
                f"🗑️ {t('workspace.delete')}",
                key="ws_delete_btn",
                disabled=not is_named_ws,
                use_container_width=True,
                help=t("workspace.delete_help") if not is_named_ws else None,
            ):
                st.session_state[STATE_SHOW_DELETE_CONFIRM] = current_ws

    # Dialogs
    _render_create_dialog(manager)
    _render_delete_confirm(manager)


def _render_create_dialog(manager: WorkspaceManager) -> None:
    """Render the create workspace dialog."""
    if not st.session_state.get(STATE_SHOW_CREATE_DIALOG):
        return

    st.markdown("---")
    st.markdown(f"**{t('workspace.create_title')}**")

    new_name = st.text_input(
        t("workspace.name"),
        placeholder="my_workspace",
        help=t("workspace.name_help"),
        key="ws_new_name",
    )

    col1, col2, _ = st.columns([1, 1, 2])

    with col1:
        if st.button(f"✓ {t('workspace.create')}", key="ws_confirm_create", type="primary"):
            if not new_name:
                st.error(t("workspace.error.name_empty"))
                return

            is_valid, error_key = manager.validate_workspace_name(new_name)
            if not is_valid:
                st.error(t(error_key))
                return

            success, error_key = manager.create_workspace(new_name)
            if success:
                set_current_workspace(new_name)
                st.session_state[STATE_SHOW_CREATE_DIALOG] = False
                st.rerun()
            else:
                st.error(t(error_key))

    with col2:
        if st.button(t("common.cancel"), key="ws_cancel_create"):
            st.session_state[STATE_SHOW_CREATE_DIALOG] = False
            st.rerun()


def _render_delete_confirm(manager: WorkspaceManager) -> None:
    """Render the delete confirmation dialog."""
    ws_to_delete = st.session_state.get(STATE_SHOW_DELETE_CONFIRM)
    if not ws_to_delete:
        return

    st.warning(f"⚠️ {t('workspace.delete_confirm', name=ws_to_delete)}")

    col1, col2, _ = st.columns([1, 1, 2])

    with col1:
        if st.button(f"🗑️ {t('workspace.delete')}", key="ws_confirm_delete", type="primary"):
            success, error_key = manager.delete_workspace(ws_to_delete)
            if success:
                # Switch to anonymous workspace
                browser_id = st.session_state.get("workspace_browser_id", "default")
                anonymous_ws = manager.get_anonymous_workspace_name(browser_id)
                set_current_workspace(anonymous_ws)
                manager.ensure_workspace_exists(anonymous_ws)
                st.session_state[STATE_SHOW_DELETE_CONFIRM] = None
                st.rerun()
            else:
                st.error(t(error_key))

    with col2:
        if st.button(t("common.cancel"), key="ws_cancel_delete"):
            st.session_state[STATE_SHOW_DELETE_CONFIRM] = None
            st.rerun()
