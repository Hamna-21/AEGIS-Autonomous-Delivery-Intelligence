"""Reusable frontend theme tokens and CSS injection."""


def inject(st):
    st.markdown("""<style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Space+Grotesk:wght@500;600;700&display=swap');
    :root { --black:#050b09; --forest:#0a1713; --panel:#10221c; --line:#24453a; --mint:#72e8bb; --emerald:#23c98b; --cyan:#6cdcf4; --amber:#ffc857; --muted:#91aaa0; }
    .stApp { background:radial-gradient(circle at 76% 5%,#15372c 0,#08130f 38%,#040806 100%); color:#e8f4ee; font-family:'DM Sans',sans-serif; }
    h1,h2,h3 { font-family:'Space Grotesk',sans-serif; letter-spacing:-.025em; }
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
    .section-card h3 { margin-top:0; }
    .grid-svg { width:100%; max-height:610px; background:#06100c; border-radius:12px; padding:.3rem; border:1px solid #285846; }
    .pill { display:inline-block; padding:.25rem .55rem; border-radius:99px; background:#12382a; color:var(--mint); font-size:.68rem; margin:.12rem; border:1px solid #285846; }
    .trace { border-left:2px solid var(--mint); padding:.35rem .7rem; margin:.4rem 0; background:#102b22; border-radius:0 8px 8px 0; }
    .stButton>button { border:1px solid #356c57; background:#12382a; color:#e8f4ee; border-radius:8px; }
    .stButton>button:hover { border-color:var(--mint); color:#fff; }
    </style>""", unsafe_allow_html=True)
