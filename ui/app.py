"""AEGIS Streamlit application shell."""
import streamlit as st
import numpy as np

from common import MEMBERS
from ui.backend import (INSTANCE, make_edges, make_problem, load_ui_grid, plan,
                        read_csv, run_search)
from ui.components import inject_styles
from ui.config import AppContext, PAGES
from ui.pages import render


def build_context():
    grid, _source = load_ui_grid()
    start = tuple(INSTANCE["start"])
    if start[0] >= grid.shape[0] or start[1] >= grid.shape[1] or grid[start] < 0:
        start = tuple(np.argwhere(grid >= 0)[0])
    stops = [tuple(stop) for stop in INSTANCE["delivery_stops"] if stop[0] < grid.shape[0] and stop[1] < grid.shape[1]]
    stops = stops or [(grid.shape[0] - 1, grid.shape[1] - 1)]
    rival = (0, min(grid.shape[1] - 1, grid.shape[0] - 1))
    if grid[rival] < 0:
        rival = (0, 0)
    return AppContext(grid, start, rival, stops)


def main():
    st.set_page_config(page_title="AEGIS", page_icon="◈", layout="wide")
    inject_styles()
    ctx = build_context()
    st.session_state.setdefault("flooded", set())
    st.sidebar.markdown('<div class="brand-mark">◈ AEGIS</div><div class="brand-sub">AUTONOMOUS DELIVERY<br>INTELLIGENCE</div>', unsafe_allow_html=True)
    st.sidebar.markdown('<span class="pill">ONLINE</span><span class="pill">LOCAL WORLD MODEL</span>', unsafe_allow_html=True)
    page = st.sidebar.radio("Pages", PAGES, format_func=lambda name: f"{name}")
    st.sidebar.caption("Team")
    for name, cms in MEMBERS:
        st.sidebar.write(f"{name} | {cms}")
    render(page, ctx)


if __name__ == "__main__":
    main()


__all__ = ["INSTANCE", "make_edges", "make_problem", "load_ui_grid", "plan",
           "read_csv", "run_search", "build_context", "main"]
