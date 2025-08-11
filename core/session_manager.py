"""
Centralized session state management for Streamlit app.
"""

import streamlit as st
from datetime import datetime

SESSION_DEFAULTS = {
    "dataset": None,
    "selected_model": None,
    "current_view": "Dataset Selection",
}


def initialize_session():
    for k, v in SESSION_DEFAULTS.items():
        if k not in st.session_state:
            st.session_state[k] = v
    # Initialize debug structures
    if "transition_log" not in st.session_state:
        st.session_state["transition_log"] = []
    if "debug_transitions" not in st.session_state:
        st.session_state["debug_transitions"] = True


def get_session(key):
    return st.session_state.get(key, SESSION_DEFAULTS.get(key))


def set_session(key, value):
    old_value = st.session_state.get(key)
    st.session_state[key] = value
    # Append transition to debug log
    try:
        if st.session_state.get("debug_transitions", True):
            ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            st.session_state.setdefault("transition_log", [])
            st.session_state["transition_log"].append(
                f"{ts} set {key}: {old_value} -> {value}"
            )
    except Exception:
        # Never let logging break app flow
        pass


def clear_transition_log():
    st.session_state["transition_log"] = []


def reset_session():
    for k in SESSION_DEFAULTS:
        st.session_state[k] = SESSION_DEFAULTS[k]
    st.session_state["transition_log"] = []
    st.rerun()
