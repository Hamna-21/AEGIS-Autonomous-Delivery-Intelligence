"""AEGIS premium mission-control UI. Run with: streamlit run ui/app.py"""
from __future__ import annotations

import html
import json
import csv
from pathlib import Path

import numpy as np

from common import MEMBERS, load_instance, stamp
from part_b.search import (DeliveryProblem, a_star, bfs, depth_limited_search,
                           greedy_best_first_search, iterative_deepening_search,
                           uniform_cost_search)
from part_c.game import ReducedDeliveryGame, alpha_beta, minimax
from part_d.interface import AEGISCommandInterface, default_grid
from world.fog import observe
from world.grid import load_grid
from world.transitions import legal_neighbors

ROOT = Path(__file__).resolve().parents[1]
INSTANCE = load_instance()
TERRAIN_COLORS = {0: "#152d3b", 1: "#31504d", 2: "#704b2b", 3: "#725b27", -1: "#070d16"}
ALGORITHM_LABELS = {"BFS": bfs, "Uniform Cost Search": uniform_cost_search,
                    "Depth-Limited Search": depth_limited_search,
                    "Iterative Deepening Search": iterative_deepening_search,
                    "Greedy Best-First": greedy_best_first_search, "A*": a_star}


def load_ui_grid():
    path = ROOT / "data" / f"grid_{INSTANCE['instance_id']}.npy"
    try:
        return load_grid(path), "Part A generated NumPy grid"
    except FileNotFoundError:
        return default_grid(), "Deterministic Part D fallback grid, Part A asset unavailable"


def make_edges(grid):
    edges = set()
    for row in range(grid.shape[0]):
        for col in range(grid.shape[1]):
            cell = (row, col)
            if grid[cell] < 0:
                continue
            for nxt in legal_neighbors(grid, cell):
                if (row + col) % 3 != 0 or nxt[0] >= row:
                    edges.add((cell, nxt))
    return edges


def make_problem(grid, start, stops, flood_enabled=False, flooded=(), one_way=False):
    edges = make_edges(grid) if one_way else None
    return DeliveryProblem(grid, tuple(start), [tuple(x) for x in stops], INSTANCE["terrain_cost"],
                           allowed_edges=edges, flooded_cells=flooded,
                           flood_enabled=flood_enabled, flood_interval=5)


def run_search(problem, label):
    if label == "Depth-Limited Search":
        return depth_limited_search(problem, 80)
    if label == "Iterative Deepening Search":
        return iterative_deepening_search(problem, 80)
    if label == "Greedy Best-First":
        return greedy_best_first_search(problem)
    return ALGORITHM_LABELS[label](problem)


def grid_svg(grid, start, rival, stops, path=(), flooded=(), fog_radius=None, title="AEGIS grid"):
    rows, cols = grid.shape
    size = max(12, min(28, 660 // max(rows, cols)))
    width, height = cols * size, rows * size
    flooded = set(flooded)
    visible = None
    if fog_radius is not None:
        rr, cc = np.ogrid[:rows, :cols]
        visible = np.maximum(abs(rr - start[0]), abs(cc - start[1])) <= fog_radius
    parts = [f'<svg class="grid-svg" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">']
    for r in range(rows):
        for c in range(cols):
            code = int(grid[r, c])
            fill = TERRAIN_COLORS.get(code, "#101b28")
            if visible is not None and not visible[r, c]:
                fill = "#0c1420"
            if (r, c) in flooded:
                fill = "#123e5d"
            parts.append(f'<rect x="{c*size}" y="{r*size}" width="{size-1}" height="{size-1}" rx="2" fill="{fill}"/>')
    for a, b in zip(path, list(path)[1:]):
        x1, y1 = a[1]*size+size//2, a[0]*size+size//2
        x2, y2 = b[1]*size+size//2, b[0]*size+size//2
        parts.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#25d9ff" stroke-width="{max(2,size//5)}" stroke-linecap="round"/>')
    for cell, color, radius in [(tuple(start), "#27e0a3", size*.30), (tuple(rival), "#ffb454", size*.28)]:
        parts.append(f'<circle cx="{cell[1]*size+size//2}" cy="{cell[0]*size+size//2}" r="{radius}" fill="{color}" stroke="#eef7ff" stroke-width="1"/>')
    for stop in stops:
        parts.append(f'<circle cx="{stop[1]*size+size//2}" cy="{stop[0]*size+size//2}" r="{size*.18}" fill="#ffc857" stroke="#fff1b5"/>')
    parts.append("</svg>")
    return "".join(parts)


def styles(st):
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --navy:#07111f; --graphite:#0d1a29; --panel:#101f31; --line:#20364b; --cyan:#25d9ff; --amber:#ffc857; --green:#27e0a3; --muted:#8aa2b9; }
    .stApp { background:radial-gradient(circle at 80% 0%,#12324b 0,#07111f 42%,#050b13 100%); color:#e8f2f8; font-family:'DM Sans',sans-serif; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.025em; }
    [data-testid="stSidebar"] { background:#081522; border-right:1px solid var(--line); }
    .hero { padding:1.2rem 1.4rem; border:1px solid #1f5267; border-radius:18px; background:linear-gradient(125deg,#10263a,#0c1726); box-shadow:0 12px 35px #02070d66; margin-bottom:1rem; }
    .eyebrow { color:var(--cyan); font-size:.72rem; text-transform:uppercase; letter-spacing:.16em; font-weight:700; }
    .hero-title { font:700 2.1rem 'Space Grotesk'; margin:.2rem 0; }
    .hero-copy { color:var(--muted); max-width:760px; }
    .metric-card { background:linear-gradient(150deg,#14263a,#0d1b2a); border:1px solid var(--line); border-radius:14px; padding:1rem; min-height:92px; box-shadow:0 8px 20px #02070d44; transition:transform .2s ease,border-color .2s ease; }
    .metric-card:hover { transform:translateY(-2px); border-color:#367894; }
    .metric-card.cyan { border-top:2px solid var(--cyan); } .metric-card.amber { border-top:2px solid var(--amber); } .metric-card.green { border-top:2px solid var(--green); }
    .metric-label { color:var(--muted); font-size:.72rem; text-transform:uppercase; letter-spacing:.1em; } .metric-value { font:700 1.55rem 'Space Grotesk'; margin-top:.3rem; }
    .section-card { border:1px solid var(--line); border-radius:16px; padding:1rem; background:#0d1b2a99; margin:.5rem 0 1rem; }
    .grid-svg { width:100%; max-height:610px; background:#081522; border-radius:12px; padding:.4rem; border:1px solid var(--line); }
    .pill { display:inline-block; padding:.3rem .6rem; border-radius:99px; background:#123449; color:var(--cyan); font-size:.75rem; margin:.15rem; }
    .trace { border-left:2px solid var(--cyan); padding:.35rem .7rem; margin:.4rem 0; background:#10263a; border-radius:0 8px 8px 0; }
    .stButton>button { border:1px solid #28647a; background:#102c40; color:#e8f2f8; border-radius:9px; } .stButton>button:hover { border-color:var(--cyan); color:#fff; }
    </style>""", unsafe_allow_html=True)


def hero(st, eyebrow, title, copy):
    st.markdown(f'<div class="hero"><div class="eyebrow">{eyebrow}</div><div class="hero-title">{title}</div><div class="hero-copy">{copy}</div></div>', unsafe_allow_html=True)


def metric_card(st, label, value, accent="cyan"):
    st.markdown(f'<div class="metric-card {accent}"><div class="metric-label">{html.escape(label)}</div><div class="metric-value">{html.escape(str(value))}</div></div>', unsafe_allow_html=True)


def main():
    import streamlit as st
    st.set_page_config(page_title="AEGIS Mission Control", page_icon="◈", layout="wide", initial_sidebar_state="expanded")
    styles(st)
    grid, grid_source = load_ui_grid()
    start = tuple(INSTANCE["start"]) if all(x < grid.shape[0] for x in INSTANCE["start"]) else (0, 0)
    stops = [tuple(x) for x in INSTANCE["delivery_stops"] if all(y < grid.shape[0] for y in x)] or [(grid.shape[0]-1, grid.shape[1]-1)]
    if grid[start] < 0:
        start = next((tuple(x) for x in np.argwhere(grid >= 0)), (0, 0))
    rival = (0, min(grid.shape[1]-1, grid.shape[0]-1))
    if grid[rival] < 0:
        rival = (0, 0)
    st.session_state.setdefault("route", [])
    st.session_state.setdefault("flooded", set())
    st.session_state.setdefault("last_result", None)
    st.sidebar.markdown("# ◈ AEGIS")
    st.sidebar.caption("Autonomous Environmental Guidance & Intelligence System")
    st.sidebar.markdown('<span class="pill">ONLINE</span><span class="pill">LOCAL WORLD MODEL</span>', unsafe_allow_html=True)
    page = st.sidebar.radio("Mission modules", ["Mission Control", "Algorithm Laboratory", "Rival Intelligence", "Environment Laboratory", "AI Command Centre", "Research Analytics", "System Architecture"])
    st.sidebar.divider()
    st.sidebar.caption(f"Instance {INSTANCE['instance_id']} · {grid_source}")
    st.sidebar.caption("Team")
    for name, cms in MEMBERS:
        st.sidebar.write(f"{name} · {cms}")
    if page == "Mission Control": mission_control(st, grid, start, rival, stops, grid_source)
    elif page == "Algorithm Laboratory": algorithm_lab(st, grid, start, stops)
    elif page == "Rival Intelligence": rival_intelligence(st, grid, start, rival, stops)
    elif page == "Environment Laboratory": environment_lab(st, grid, start, rival, stops)
    elif page == "AI Command Centre": command_centre(st, grid)
    elif page == "Research Analytics": analytics(st, grid, start, stops)
    else: architecture(st)


def mission_control(st, grid, start, rival, stops, source):
    hero(st, "MISSION CONTROL / LIVE OPERATIONS", "Autonomous delivery mission", "Monitor the generated world, dispatch a real route, and inspect the current agent state.")
    c1,c2,c3,c4 = st.columns(4)
    metric_card(c1, "Agent position", start)
    metric_card(c2, "Rival position", rival, "amber")
    metric_card(c3, "Delivery stops", len(stops), "amber")
    metric_card(c4, "Grid source", "Part A" if "Part A" in source else "Fallback", "green")
    left, right = st.columns([1.45, 1])
    with left:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Tactical map")
        st.markdown(grid_svg(grid, start, rival, stops, st.session_state.route, st.session_state.flooded), unsafe_allow_html=True)
        st.caption("Cyan route · emerald agent · amber rival · gold delivery stop · blue flood")
        st.markdown('</div>', unsafe_allow_html=True)
    with right:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.subheader("Dispatch controls")
        algorithm = st.selectbox("Planner", list(ALGORITHM_LABELS), index=5, key="mission_alg")
        flood = st.checkbox("Flood effects enabled", key="mission_flood")
        one_way = st.checkbox("Enforce generated one-way streets", key="mission_oneway")
        if st.button("Plan live route", type="primary", use_container_width=True):
            result = run_search(make_problem(grid, start, stops, flood, st.session_state.flooded, one_way), algorithm)
            st.session_state.route = result.path
            st.session_state.last_result = result
        result = st.session_state.last_result
        if result:
            st.success("Route acquired" if result.found else "No route found")
            metric_card(st, "Solution cost", result.cost if result.found else "N/A", "green")
            st.write({"nodes_expanded": result.nodes_expanded, "runtime_seconds": round(result.runtime_seconds, 6), "memory": result.max_memory_states})
        else:
            st.info("Awaiting dispatch")
        st.markdown('</div>', unsafe_allow_html=True)


def algorithm_lab(st, grid, start, stops):
    hero(st, "ALGORITHM LABORATORY / PART B", "Search performance lab", "Select any implemented planner and inspect the real explored-state and route metrics.")
    algorithm = st.selectbox("Algorithm", list(ALGORITHM_LABELS), key="lab_alg")
    flood = st.checkbox("Enable time-dependent flood", key="lab_flood")
    if st.button("Execute algorithm", type="primary"):
        st.session_state.lab_result = run_search(make_problem(grid, start, stops, flood), algorithm)
    result = st.session_state.get("lab_result")
    if result:
        cols = st.columns(5)
        for col, label, value in zip(cols, ["Found", "Cost", "Expanded", "Runtime", "Memory"], [result.found, result.cost, result.nodes_expanded, f"{result.runtime_seconds:.5f}s", result.max_memory_states]):
            metric_card(col, label, value, "green" if label == "Found" else "cyan")
        st.markdown(grid_svg(grid, start, stops[-1], stops, result.path), unsafe_allow_html=True)
        st.dataframe({"step": list(range(len(result.path))), "row": [x[0] for x in result.path], "col": [x[1] for x in result.path]}, use_container_width=True)
    else:
        st.info("Execute a planner to populate the laboratory")


def rival_intelligence(st, grid, start, rival, stops):
    hero(st, "RIVAL INTELLIGENCE / PART C", "Adversarial decision theatre", "Compare exhaustive Minimax reasoning with Alpha-Beta pruning and inspect the rival-aware candidate set.")
    depth = st.slider("Search depth", 2, 6, int(INSTANCE.get("adversary_search_depth", 6)))
    algorithm = st.radio("Decision engine", ["Alpha-Beta", "Minimax"], horizontal=True)
    passable = {tuple(x) for x in np.argwhere(grid >= 0)}
    target_list = [tuple(x) for x in stops if tuple(x) in passable] or [(grid.shape[0]-1, grid.shape[1]-1)]
    game = ReducedDeliveryGame(passable, start, rival, target_list, horizon=max(12, depth + 4))
    decision = alpha_beta(game, depth=depth) if algorithm == "Alpha-Beta" else minimax(game, depth=depth)
    cols = st.columns(4)
    for col, label, value, accent in [(cols[0], "Selected move", decision.move, "green"), (cols[1], "Value", round(decision.value, 2), "cyan"), (cols[2], "Nodes", decision.stats.nodes, "cyan"), (cols[3], "Cutoffs", decision.stats.cutoffs, "amber")]:
        metric_card(col, label, value, accent)
    st.markdown(grid_svg(grid, start, rival, target_list), unsafe_allow_html=True)
    st.subheader("Candidate moves")
    st.bar_chart({"move": [x[0] for x in decision.candidates], "value": [x[1] for x in decision.candidates]}, x="move", y="value")
    st.caption("Candidate values and the selected move come directly from Part C game-tree search.")


def environment_lab(st, grid, start, rival, stops):
    hero(st, "ENVIRONMENT LABORATORY / PART A", "World-state controls", "Expose the environmental variables that alter sensing, movement cost, and execution risk.")
    c1,c2 = st.columns(2)
    fog_radius = c1.slider("Fog visibility radius", 0, max(grid.shape), 3)
    slip = c2.slider("Slip probability", 0.0, 1.0, float(INSTANCE["slip_probability"]), 0.01)
    flood_on = st.toggle("Flood enabled", value=False)
    one_way = st.toggle("One-way streets enabled", value=False)
    flood_cell = st.selectbox("Seed flood cell", [tuple(x) for x in np.argwhere(grid >= 0)], format_func=lambda x: str(x))
    if st.button("Spread flood one step"):
        from world.flood import spread_flood
        _, flooded = spread_flood(grid, {flood_cell})
        st.session_state.flooded = set(flooded)
    observed, mask = observe(grid, start, fog_radius)
    st.write({"visible_cells": int(mask.sum()), "slip_probability": slip, "flooded_cells": len(st.session_state.flooded), "one_way": one_way, "flood_enabled": flood_on})
    st.markdown(grid_svg(grid, start, rival, stops, flooded=st.session_state.flooded, fog_radius=fog_radius), unsafe_allow_html=True)
    st.dataframe(observed, use_container_width=True)


def command_centre(st, grid):
    hero(st, "AI COMMAND CENTRE / PART D", "Natural-language mission interface", "Offline parser mode is active. Commands are validated, dispatched to real tools, and traced as Sense, Reason, Act events.")
    if "command_interface" not in st.session_state:
        st.session_state.command_interface = AEGISCommandInterface(grid=grid, log_path=ROOT / "outputs" / "ui_part_d_events.jsonl")
    interface = st.session_state.command_interface
    st.info(f"Mode: {interface.mode}. No API key is required and this parser is not a real LLM.")
    request = st.text_area("Command", "Plan a route from (0,0) to (6,6) using A*")
    if st.button("Execute command", type="primary"):
        try:
            st.session_state.command_result = interface.handle(request)
            st.session_state.command_error = None
        except Exception as exc:
            st.session_state.command_error = str(exc)
    if st.session_state.get("command_error"):
        st.error(st.session_state.command_error)
    if st.session_state.get("command_result"):
        result = st.session_state.command_result
        st.success("Validated tool result")
        st.json(result.get("result", result))
        st.subheader("Sense → Reason → Act trace")
        for event in result.get("trace", []):
            st.markdown(f'<div class="trace"><b>{event["phase"]}</b> · {event["name"]}<br><small>{html.escape(json.dumps(event["observation"]))}</small></div>', unsafe_allow_html=True)


def analytics(st, grid, start, stops):
    hero(st, "RESEARCH ANALYTICS / PARTS B-D", "Evidence dashboard", "Charts below are computed from live calls into the implemented planners and command interface.")
    if st.button("Run analytics snapshot", type="primary") or "analytics" not in st.session_state:
        values = []
        for label in ALGORITHM_LABELS:
            result = run_search(make_problem(grid, start, stops), label)
            values.append({"algorithm": label, "cost": result.cost, "nodes": result.nodes_expanded, "runtime_ms": result.runtime_seconds*1000})
        st.session_state.analytics = values
    values = st.session_state.analytics
    st.subheader("Part B comparison")
    st.bar_chart({"algorithm": [x["algorithm"] for x in values], "nodes": [x["nodes"] for x in values]}, x="algorithm", y="nodes")
    st.caption(stamp())
    st.dataframe(values, use_container_width=True)
    st.subheader("Part C depth profile")
    passable = {tuple(x) for x in np.argwhere(grid >= 0)}
    game = ReducedDeliveryGame(passable, start, (0, min(grid.shape[1]-1, grid.shape[0]-1)), stops[:1], horizon=14)
    rows = []
    for depth in range(2, 7):
        mm, ab = minimax(game, depth=depth), alpha_beta(game, depth=depth)
        rows.append({"depth": depth, "minimax_nodes": mm.stats.nodes, "alpha_beta_nodes": ab.stats.nodes, "pruning_percent": 100*(mm.stats.nodes-ab.stats.nodes)/max(1,mm.stats.nodes)})
    st.line_chart({"depth": [x["depth"] for x in rows], "Minimax": [x["minimax_nodes"] for x in rows], "Alpha-Beta": [x["alpha_beta_nodes"] for x in rows]}, x="depth", y=["Minimax", "Alpha-Beta"])
    st.caption(stamp())
    st.dataframe(rows, use_container_width=True)
    st.subheader("Part D command metrics")
    metrics_path = ROOT / "outputs" / "research" / "part_d_llm_raw.csv"
    if metrics_path.exists():
        with metrics_path.open(encoding="utf-8") as stream:
            command_rows = list(csv.DictReader(stream))
        successes = sum(row.get("success") == "True" for row in command_rows)
        matches = sum(row.get("direct_planner_match") == "True" for row in command_rows)
        command_metrics = {"mode": command_rows[0].get("mode", "offline_parser") if command_rows else "offline_parser",
                           "evaluation_requests": len(command_rows),
                           "success_rate": successes / len(command_rows) if command_rows else 0.0,
                           "direct_planner_matches": f"{matches}/{len(command_rows)}"}
    else:
        command_metrics = {"mode": "offline_parser", "status": "Run python run_all.py to load measured Part D results"}
    st.caption(stamp())
    st.write(command_metrics)


def architecture(st):
    hero(st, "SYSTEM ARCHITECTURE", "AEGIS operating picture", "A modular pipeline from world sensing to search, adversarial reasoning, command tools, and research evidence.")
    st.code("""World model (Part A)\n        ↓\nSearch planners (Part B) ↔ Rival intelligence (Part C)\n        ↓                         ↓\nCommand tools and event trace (Part D)\n        ↓\nStreamlit UI and research analytics""", language="text")
    for title, copy in [("Part A / World model", "Terrain, obstacles, fog, flood, movement and one-way constraints."), ("Part B / Route planning", "Six from-scratch planners with multi-delivery state and metrics."), ("Part C / Rival intelligence", "Minimax, Alpha-Beta, move ordering, stochastic execution and replanning."), ("Part D / Command centre", "Validated offline natural-language routing, simulation, flood tool and event traces."), ("Research layer", "Live planner runs, rival depth profiles, command metrics and confidence reporting.")]:
        st.markdown(f'<div class="section-card"><h3>{title}</h3><p>{copy}</p></div>', unsafe_allow_html=True)
    st.subheader("Team and CMS IDs")
    st.table({"Member": [name for name, _ in MEMBERS], "CMS ID": [cms for _, cms in MEMBERS]})


if __name__ == "__main__":
    main()
