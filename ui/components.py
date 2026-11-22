"""Reusable visual components for the Streamlit frontend."""
import html
import streamlit as st

COLORS = {0: "#1e4a35", 1: "#2f6b4a", 2: "#6b5a2f", 3: "#7a6a2a", -1: "#050b08"}


def inject_styles():
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --black:#050b09; --forest:#0a1713; --panel:#10221c; --line:#24453a; --mint:#72e8bb; --cyan:#6cdcf4; --amber:#ffc857; --muted:#91aaa0; }
    .stApp { background:radial-gradient(circle at 78% 0%,#143227 0,#08130f 38%,#040806 100%); color:#e8f4ee; font-family:'DM Sans',sans-serif; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.02em; }
    [data-testid="stSidebar"] { background:#07110d; border-right:1px solid #1e3d32; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label { border-radius:8px; padding:.4rem .6rem; color:#91aaa0; }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover { background:#10271f; color:#e8f4ee; }
    .brand-mark { color:var(--mint); font:700 1.35rem 'Space Grotesk'; letter-spacing:.07em; }
    .brand-sub { color:var(--muted); font-size:.65rem; line-height:1.35; }
    .hero { padding:1rem 1.25rem; border:1px solid #285846; border-radius:16px; background:#0d2119; box-shadow:0 8px 20px #00000035; margin-bottom:.8rem; }
    .eyebrow { color:var(--mint); font-size:.68rem; text-transform:uppercase; letter-spacing:.16em; font-weight:700; }
    .hero-title { font:700 1.75rem 'Space Grotesk'; margin:.15rem 0; }
    .hero-copy { color:var(--muted); max-width:760px; font-size:.9rem; }
    .metric-card { background:#10271f; border:1px solid var(--line); border-radius:12px; padding:.8rem; min-height:74px; box-shadow:0 6px 16px #0000002b; }
    .metric-card.cyan { border-top:2px solid var(--cyan); } .metric-card.amber { border-top:2px solid var(--amber); } .metric-card.green { border-top:2px solid var(--mint); }
    .metric-label { color:var(--muted); font-size:.67rem; text-transform:uppercase; letter-spacing:.1em; } .metric-value { font:700 1.35rem 'Space Grotesk'; margin-top:.25rem; }
    .section-card { border:1px solid var(--line); border-radius:14px; padding:.9rem; background:#0d2119; margin:.45rem 0 .8rem; box-shadow:0 8px 20px #00000024; }
    .grid-svg { width:100%; max-height:600px; background:#06100c; border-radius:12px; padding:.3rem; border:1px solid #285846; }
    .pill { display:inline-block; padding:.25rem .55rem; border-radius:99px; background:#12382a; color:var(--mint); font-size:.68rem; margin:.12rem; border:1px solid #285846; }
    .trace { border-left:2px solid var(--mint); padding:.35rem .7rem; margin:.4rem 0; background:#102b22; border-radius:0 8px 8px 0; }
    .stButton>button { border:1px solid #356c57; background:#12382a; color:#e8f4ee; border-radius:8px; }
    .stButton>button:hover { border-color:var(--mint); color:#fff; }
    </style>""", unsafe_allow_html=True)


def hero(eyebrow, title, copy):
    st.markdown(f'<div class="hero"><div class="eyebrow">{html.escape(eyebrow)}</div><div class="hero-title">{html.escape(title)}</div><div class="hero-copy">{html.escape(copy)}</div></div>', unsafe_allow_html=True)


def metric_card(label, value, accent="cyan"):
    st.markdown(f'<div class="metric-card {accent}"><div class="metric-label">{html.escape(label)}</div><div class="metric-value">{html.escape(str(value))}</div></div>', unsafe_allow_html=True)


def draw(grid, start, rival, stops, path=(), flooded=(), fog=None):
    rows, cols = grid.shape; size = max(12, min(28, 660 // max(rows, cols)))
    mid = lambda cell: (cell[1] * size + size // 2, cell[0] * size + size // 2)
    svg = [f'<svg class="grid-svg" viewBox="0 0 {cols * size} {rows * size}">']
    for r in range(rows):
        for c in range(cols):
            color = COLORS.get(int(grid[r, c]), "#222")
            if fog is not None and max(abs(r - start[0]), abs(c - start[1])) > fog: color = "#0a120e"
            if (r, c) in flooded: color = "#1d6fa5"
            svg.append(f'<rect x="{c * size}" y="{r * size}" width="{size - 1}" height="{size - 1}" fill="{color}"/>')
    for a, b in zip(path, list(path)[1:]):
        (x1, y1), (x2, y2) = mid(a), mid(b); svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="#7dffb0" stroke-width="3"/>')
    for cell, color in [(start, "#fff"), (rival, "#ffb454")] + [(p, "#ffd84d") for p in stops]:
        x, y = mid(cell); svg.append(f'<circle cx="{x}" cy="{y}" r="{size * 0.25}" fill="{color}"/>')
    svg.append("</svg>"); st.markdown("".join(svg), unsafe_allow_html=True)
    st.caption("Green route | white agent | orange rival | yellow stops | blue flood")
