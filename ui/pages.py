"""Streamlit page renderers. Each page calls existing AEGIS backends."""
import streamlit as st
import numpy as np

from common import MEMBERS, stamp
from part_c.game import ReducedDeliveryGame, alpha_beta, minimax
from part_d.interface import AEGISCommandInterface
from world.fog import observe
from world.flood import spread_flood

from .backend import INSTANCE, PLANNERS, ROOT, plan, read_csv
from .components import draw, hero, metric_card


def mission_control(ctx):
    hero("MISSION CONTROL", "Autonomous delivery mission", "Dispatch the planner and inspect the current environment.")
    a, b, c = st.columns(3); metric_card("Agent", ctx.start); metric_card("Rival", ctx.rival, "amber"); metric_card("Delivery stops", len(ctx.stops), "amber")
    name = st.selectbox("Planner", list(PLANNERS), index=5, key="mission_planner")
    flood = st.checkbox("Flood effects", key="mission_flood")
    one_way = st.checkbox("One-way streets", key="mission_one_way")
    if st.button("Plan route", type="primary", key="mission_plan"):
        st.session_state.result = plan(ctx.grid, ctx.start, ctx.stops, name, flood, st.session_state.flooded, one_way)
    result = st.session_state.get("result")
    if result:
        st.success("Route found" if result.found else "No route found")
        st.write({"cost": result.cost, "nodes_expanded": result.nodes_expanded, "runtime_seconds": round(result.runtime_seconds, 6), "memory": result.max_memory_states})
    draw(ctx.grid, ctx.start, ctx.rival, ctx.stops, result.path if result else (), st.session_state.flooded)


def assistant(ctx):
    hero("AI ASSISTANT", "Validated command interface", "Use natural-language delivery commands during the demonstration.")
    if "interface" not in st.session_state:
        st.session_state.interface = AEGISCommandInterface(grid=ctx.grid, log_path=ROOT / "outputs" / "ui_part_d_events.jsonl")
    interface = st.session_state.interface
    st.info(f"Mode: {interface.mode}. Set GOOGLE_API_KEY to enable Gemini natural-language commands.")
    command = st.text_area("Command", "Plan a route from (0,0) to (6,6) using A*")
    if st.button("Execute", type="primary"):
        try:
            result = interface.handle(command); st.json(result.get("result", result))
            st.subheader("Sense, Reason, Act trace")
            for event in result.get("trace", []): st.write(f"**{event['phase']}**: {event['name']}"); st.caption(str(event["observation"]))
        except Exception as error: st.error(str(error))


def algorithm_lab(ctx):
    hero("ALGORITHM LAB", "Search performance", "Compare all six planners against the same current world.")
    name = st.selectbox("Algorithm", list(PLANNERS), key="lab_planner"); flood = st.checkbox("Time-dependent flood", key="lab_flood")
    if st.button("Run", type="primary", key="lab_run"):
        r = plan(ctx.grid, ctx.start, ctx.stops, name, flood); cols = st.columns(4)
        cols[0].metric("Found", str(r.found)); cols[1].metric("Cost", r.cost); cols[2].metric("Expanded", r.nodes_expanded); cols[3].metric("Runtime", f"{r.runtime_seconds:.5f}s")
        draw(ctx.grid, ctx.start, ctx.rival, ctx.stops, r.path); st.dataframe([{"step": i, "row": p[0], "col": p[1]} for i, p in enumerate(r.path)], use_container_width=True)


def rival_page(ctx):
    hero("RIVAL INTELLIGENCE", "Adversarial decision theatre", "Inspect Minimax and Alpha-Beta choices in the reduced game.")
    depth = st.slider("Search depth", 2, 6, int(INSTANCE.get("adversary_search_depth", 6))); engine = st.radio("Engine", ["Alpha-Beta", "Minimax"], horizontal=True)
    passable = {tuple(x) for x in np.argwhere(ctx.grid >= 0)}; targets = [s for s in ctx.stops if s in passable] or [(ctx.grid.shape[0]-1, ctx.grid.shape[1]-1)]
    game = ReducedDeliveryGame(passable, ctx.start, ctx.rival, targets, horizon=max(12, depth + 4)); decision = alpha_beta(game, depth=depth) if engine == "Alpha-Beta" else minimax(game, depth=depth)
    cols = st.columns(4); cols[0].metric("Move", str(decision.move)); cols[1].metric("Value", round(decision.value, 2)); cols[2].metric("Nodes", decision.stats.nodes); cols[3].metric("Cutoffs", decision.stats.cutoffs)
    draw(ctx.grid, ctx.start, ctx.rival, targets); st.bar_chart({"move": [str(c[0]) for c in decision.candidates], "value": [c[1] for c in decision.candidates]}, x="move", y="value")


def environment_page(ctx):
    hero("ENVIRONMENT LAB", "World controls", "Inspect fog, flood, and execution risk without changing the world model.")
    fog = st.slider("Fog visibility radius", 0, max(ctx.grid.shape), 3); slip = st.slider("Slip probability", 0.0, 1.0, float(INSTANCE["slip_probability"]), 0.01)
    c1, c2 = st.columns(2); row = c1.number_input("Flood seed row", 0, ctx.grid.shape[0]-1, 0); col = c2.number_input("Flood seed col", 0, ctx.grid.shape[1]-1, 0)
    if st.button("Spread flood one step", key="flood_step"):
        st.session_state.flooded = set(spread_flood(ctx.grid, {(int(row), int(col))})[1])
    observed, mask = observe(ctx.grid, ctx.start, fog); st.write({"visible_cells": int(mask.sum()), "slip_probability": slip, "flooded_cells": len(st.session_state.flooded)})
    draw(ctx.grid, ctx.start, ctx.rival, ctx.stops, flooded=st.session_state.flooded, fog=fog); st.dataframe(observed, use_container_width=True)


def analytics(ctx):
    hero("RESEARCH ANALYTICS", "Measured evidence", "Run the current planners and inspect the research outputs.")
    if st.button("Run snapshot", key="analytics_run") or "analytics" not in st.session_state:
        st.session_state.analytics = [{"algorithm": name, "cost": (r := plan(ctx.grid, ctx.start, ctx.stops, name)).cost, "nodes": r.nodes_expanded, "runtime_ms": r.runtime_seconds * 1000} for name in PLANNERS]
    values = st.session_state.analytics; st.subheader("Part B comparison"); st.bar_chart({"algorithm": [v["algorithm"] for v in values], "nodes": [v["nodes"] for v in values]}, x="algorithm", y="nodes"); st.dataframe(values, use_container_width=True)
    st.subheader("Part C depth profile"); passable = {tuple(x) for x in np.argwhere(ctx.grid >= 0)}; game = ReducedDeliveryGame(passable, ctx.start, (0, min(ctx.grid.shape[1]-1, ctx.grid.shape[0]-1)), ctx.stops[:1], horizon=14); rows = []
    for depth in range(2, 7):
        mm, ab = minimax(game, depth=depth), alpha_beta(game, depth=depth); rows.append({"depth": depth, "Minimax": mm.stats.nodes, "Alpha-Beta": ab.stats.nodes})
    st.line_chart(rows, x="depth", y=["Minimax", "Alpha-Beta"])
    path = ROOT / "outputs" / "research" / "part_d_llm_raw.csv"
    if path.exists():
        data = read_csv(path); ok = sum(r.get("success") == "True" for r in data); st.write({"requests": len(data), "success_rate": ok / len(data) if data else 0})
    else: st.info("Run python run_all.py to load Part D results")
    st.caption(stamp())


def reports(_ctx=None):
    hero("REPORTS", "Research archive", "Download measured experiment data and inspect the generated summary table.")
    research = ROOT / "outputs" / "research"; files = sorted(research.glob("*.csv")) if research.exists() else []
    if not files: st.warning("No research outputs found. Run python run_all.py first.")
    for file in files: st.download_button(f"Download {file.name}", file.read_bytes(), file.name, "text/csv", key=f"report_{file.name}")
    summary = research / "summary_tables.csv"
    if summary.exists(): st.subheader("Summary tables"); st.dataframe(read_csv(summary), use_container_width=True)
    st.caption(stamp())


def architecture(_ctx=None):
    hero("SYSTEM ARCHITECTURE", "AEGIS module map", "A compact view of the research system boundaries.")
    st.code("Part A: World model\n   |\nPart B: Search planners <-> Part C: Rival intelligence\n   |\nPart D: Command tools\n   |\nStreamlit frontend", language="text")
    st.write("**Part A**: terrain, fog, flood and one-way streets"); st.write("**Part B**: six search planners"); st.write("**Part C**: Minimax and Alpha-Beta rival"); st.write("**Part D**: natural-language command interface")
    st.table({"Member": [n for n, _ in MEMBERS], "CMS ID": [c for _, c in MEMBERS]})


def settings(_ctx=None):
    hero("SETTINGS", "System profile", "Small presentation settings for the university demonstration.")
    st.toggle("Compact density", key="compact"); st.toggle("Show diagnostic labels", value=True, key="diagnostics"); st.write({"theme": "dark emerald", "research_seed": 20261122})


def render(page, ctx):
    {"Mission Control": mission_control, "AI Assistant": assistant,
     "Algorithm Laboratory": algorithm_lab, "Rival Intelligence": rival_page,
     "Environment Laboratory": environment_page, "Research Analytics": analytics,
     "Reports": reports, "System Architecture": architecture, "Settings": settings}[page](ctx)
