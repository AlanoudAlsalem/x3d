"""
X3D-M Inference Profiling Dashboard
Dark-themed Streamlit dashboard for comparing inference runs.
"""

import json
import os
import sys
import streamlit as st
import plotly.graph_objects as go
from pathlib import Path
from collections import Counter

# Running `python dashboard.py` executes this file without a ScriptRunContext, which
# spams warnings for every `st.*` call. If we are __main__ and not inside Streamlit,
# re-invoke via the Streamlit CLI (same as `streamlit run dashboard.py`).
if __name__ == "__main__":
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except ImportError:

        def get_script_run_ctx():
            return None

    if get_script_run_ctx() is None:
        from streamlit.web import cli as stcli

        sys.argv = ["streamlit", "run", str(Path(__file__).resolve()), *sys.argv[1:]]
        raise SystemExit(stcli.main())

# ── Theme colours (with transparency variants) ────────────────────────────
C_GREEN  = "#04CD6C"
C_BLUE   = "#019ADE"
C_PURPLE = "#7333D4"
C_YELLOW = "#FFC61F"
C_RED    = "#FF1F5B"

PALETTE = [C_GREEN, C_BLUE, C_PURPLE, C_YELLOW, C_RED]

BG_DARK      = "#0E1117"
BG_CARD      = "#161B22"
BG_SURFACE   = "#1C2333"
TEXT_PRIMARY  = "#E6EDF3"
TEXT_MUTED    = "#8B949E"
BORDER        = "#30363D"

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="X3D-M Profiling Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
    /* Global dark overrides */
    .stApp {{
        background-color: {BG_DARK};
        color: {TEXT_PRIMARY};
    }}
    section[data-testid="stSidebar"] {{
        background-color: {BG_CARD};
        border-right: 1px solid {BORDER};
    }}
    section[data-testid="stSidebar"] * {{
        color: {TEXT_PRIMARY} !important;
    }}

    /* Metric cards */
    .metric-card {{
        background: {BG_CARD};
        border: 1px solid {BORDER};
        border-radius: 10px;
        padding: 1.2rem 1.4rem;
        text-align: center;
    }}
    .metric-value {{
        font-size: 2rem;
        font-weight: 700;
        color: {C_GREEN};
        margin: 0.25rem 0;
    }}
    .metric-label {{
        font-size: 0.82rem;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }}

    /* Category info cards */
    .cat-card {{
        background: {BG_SURFACE};
        border: 1px solid {BORDER};
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-bottom: 0.6rem;
    }}
    .cat-card h4 {{
        margin: 0 0 0.4rem 0;
        font-size: 1rem;
    }}
    .cat-card p {{
        margin: 0;
        font-size: 0.85rem;
        color: {TEXT_MUTED};
        line-height: 1.5;
    }}

    /* Plotly chart container */
    .stPlotlyChart {{
        background: transparent !important;
    }}

    /* Hide default streamlit branding */
    #MainMenu, footer {{visibility: hidden;}}

    /* Expander styling */
    .streamlit-expanderHeader {{
        background-color: {BG_SURFACE} !important;
        border-radius: 8px;
        color: {TEXT_PRIMARY} !important;
    }}
    div[data-testid="stExpander"] details {{
        border: 1px solid {BORDER} !important;
        border-radius: 8px;
    }}
</style>
""", unsafe_allow_html=True)


# ── Data loading ───────────────────────────────────────────────────────────
STATS_DIR = Path(__file__).parent / "run_stats"


def make_run_label(meta: dict) -> str:
    """Derive a human-readable label from run metadata."""
    pinfo = meta.get("platform_info") or {}
    if not isinstance(pinfo, dict):
        pinfo = {}
    platform = pinfo.get("device_type", "unknown")
    notes = str(meta.get("notes") or "")
    notes_lower = notes.lower()

    if "c implementation" in notes_lower or "native" in notes_lower:
        method = "Native C"
    elif "single" in notes_lower:
        method = "Single-thread OpenCV"
    elif "multithread" in notes_lower or "multi-thread" in notes_lower or "multi thread" in notes_lower:
        method = "Multi-thread OpenCV"
    else:
        method = "OpenCV"

    return f"{platform}  ·  {method}"


@st.cache_data
def load_runs():
    runs = []
    for fp in sorted(STATS_DIR.glob("*.json")):
        with open(fp) as f:
            data = json.load(f)
        data["_label"] = make_run_label(data)
        data["_file"] = fp.name
        runs.append(data)
    return runs


runs = load_runs()

if not runs:
    st.error("No JSON run files found in `run_stats/`.")
    st.stop()


# ── Sidebar – run selector ─────────────────────────────────────────────────
st.sidebar.markdown(f"### Run Selection")
st.sidebar.markdown(f"<p style='color:{TEXT_MUTED}; font-size:0.82rem;'>Toggle runs to compare in the latency charts.</p>", unsafe_allow_html=True)

enabled = {}
for r in runs:
    enabled[r["_file"]] = st.sidebar.checkbox(r["_label"], value=True, key=r["_file"])

active_runs = [r for r in runs if enabled[r["_file"]]]

# ── Shared Plotly layout ──────────────────────────────────────────────────
# Only keys that are not overridden per chart — no xaxis/yaxis/margin (those collide with ** unpack).
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRIMARY, family="Inter, system-ui, sans-serif"),
    legend=dict(
        bgcolor="rgba(0,0,0,0)",
        font=dict(color=TEXT_PRIMARY, size=12),
        orientation="h",
        yanchor="bottom",
        y=1.04,
        xanchor="center",
        x=0.5,
    ),
)


# ── Header ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="margin-bottom:1.5rem;">
    <h1 style="margin:0; font-size:1.8rem; color:{TEXT_PRIMARY};">X3D-M  Inference Profiling</h1>
    <p style="margin:0.3rem 0 0 0; color:{TEXT_MUTED}; font-size:0.92rem;">
        Comparing {len(runs)} profiling runs across platforms and convolution methods
    </p>
</div>
""", unsafe_allow_html=True)


# ── Section 1 – General statistics ─────────────────────────────────────────
ref = runs[0]  # all runs share the same architecture
total_layers = len(ref["layers"])
layer_types = [l["layer_type"] for l in ref["layers"]]
type_counts = Counter(layer_types)
total_params = ref.get("total_params", sum(l["params"] for l in ref["layers"]))
total_flops = ref.get("total_flops", sum(l["flops"] for l in ref["layers"]))

cols = st.columns(4)
metrics = [
    ("Total Layers", str(total_layers), C_GREEN),
    ("Layer Categories", str(len(type_counts)), C_BLUE),
    ("Parameters", f"{total_params / 1e6:.2f} M", C_PURPLE),
    ("FLOPs", f"{total_flops / 1e9:.2f} G", C_YELLOW),
]
for col, (label, value, color) in zip(cols, metrics):
    col.markdown(f"""
    <div class="metric-card">
        <div class="metric-label">{label}</div>
        <div class="metric-value" style="color:{color};">{value}</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

# ── Section 2 – Layers by category ─────────────────────────────────────────
st.markdown(f"<h3 style='color:{TEXT_PRIMARY}; margin-bottom:0.5rem;'>Layers by Category</h3>", unsafe_allow_html=True)

# Bar chart of layer counts by type
sorted_types = sorted(type_counts.items(), key=lambda x: -x[1])
cat_names = [t[0] for t in sorted_types]
cat_counts = [t[1] for t in sorted_types]

# Assign colours to categories
def cat_color(name: str) -> str:
    n = name.lower()
    if "conv" in n:
        return C_GREEN
    if "batch" in n or "norm" in n:
        return C_BLUE
    if "relu" in n or "silu" in n or "sigmoid" in n:
        return C_PURPLE
    if "pool" in n:
        return C_YELLOW
    return TEXT_MUTED


cat_colors = [cat_color(n) for n in cat_names]

fig_cats = go.Figure(go.Bar(
    x=cat_counts,
    y=cat_names,
    orientation="h",
    marker_color=cat_colors,
    text=cat_counts,
    textposition="outside",
    textfont=dict(color=TEXT_PRIMARY, size=12),
))
fig_cats.update_layout(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor=BG_CARD,
    font=dict(color=TEXT_PRIMARY, family="Inter, system-ui, sans-serif"),
    height=max(340, len(cat_names) * 30),
    yaxis=dict(autorange="reversed", gridcolor=BORDER, zerolinecolor=BORDER),
    xaxis=dict(title="Count", gridcolor=BORDER, zerolinecolor=BORDER),
    showlegend=False,
    margin=dict(l=200, r=60, t=20, b=40),
)
st.plotly_chart(fig_cats)

# ── Info cards for each category ────────────────────────────────────────────
CATEGORY_INFO = {
    "Conv3d": {
        "title": "Conv3d — Standard 3D Convolution",
        "desc": (
            "A standard 3D convolution slides a small learnable filter (kernel) across three dimensions "
            "— width, height, and time — of the input video tensor. At each position, it computes a "
            "dot product between the kernel weights and the overlapping input values, producing one output "
            "number. This is repeated across every spatial and temporal position to build a complete output "
            "feature map. It's the most general form of convolution in this network and is used in the stem "
            "to process raw video frames."
        ),
    },
    "Conv3d (1x1x1)": {
        "title": "Conv3d 1×1×1 — Pointwise Convolution",
        "desc": (
            "A pointwise convolution uses a tiny 1×1×1 kernel, meaning it looks at only a single pixel "
            "at a single time step. Instead of capturing spatial or temporal patterns, it mixes information "
            "across channels — think of it as a learned weighted average of all the feature channels at each "
            "location. This is used to expand or compress the number of channels cheaply (e.g. from 24 → 54 "
            "or 216 → 96) without any spatial computation, making the network much more efficient."
        ),
    },
    "Conv3d (depthwise)": {
        "title": "Conv3d Depthwise — Temporal Depthwise Convolution",
        "desc": (
            "A depthwise convolution applies a separate filter to each input channel independently, rather "
            "than mixing channels together. In X3D-M, these use a 5×1×1 kernel in the stem, meaning they "
            "look at 5 consecutive time frames but only a single spatial pixel. This captures temporal motion "
            "patterns (how things change over time) very efficiently, using far fewer parameters than a "
            "standard convolution — only one filter per channel instead of one filter per input-output "
            "channel pair."
        ),
    },
    "Conv3d (3x3x3 depthwise)": {
        "title": "Conv3d 3×3×3 Depthwise — Spatial-Temporal Depthwise",
        "desc": (
            "This is a depthwise convolution with a 3×3×3 kernel — it looks at a 3-pixel neighborhood in "
            "height, width, and time simultaneously, but still processes each channel independently. It's "
            "the main spatial-temporal feature extractor in each bottleneck block, capturing local motion "
            "and texture patterns. Because it's depthwise, it uses dramatically fewer multiply-add operations "
            "than a standard 3×3×3 convolution (proportional to channels rather than channels squared)."
        ),
    },
    "BatchNorm3d": {
        "title": "BatchNorm3d — Batch Normalization",
        "desc": (
            "Batch normalization standardizes the output of a layer so that it has approximately zero mean "
            "and unit variance. During inference, it uses precomputed running statistics (mean and variance "
            "from training) to shift and scale each channel. This helps the network train faster and makes "
            "the output more stable. Computationally it's very lightweight — just a multiply and add per element."
        ),
    },
    "ReLU": {
        "title": "ReLU — Rectified Linear Unit",
        "desc": (
            "ReLU is the simplest activation function: it outputs the input directly if it's positive, "
            "or zero if it's negative. Mathematically: ReLU(x) = max(0, x). It introduces non-linearity "
            "into the network, which is essential for learning complex patterns. Without activations, "
            "stacking layers would be equivalent to a single linear transformation."
        ),
    },
    "SiLU": {
        "title": "SiLU — Sigmoid Linear Unit (Swish)",
        "desc": (
            "SiLU (also known as Swish) is a smooth activation function defined as SiLU(x) = x × sigmoid(x). "
            "Unlike ReLU which has a hard cutoff at zero, SiLU allows small negative values through, "
            "which can help gradient flow during training. It's used after the depthwise convolution "
            "in each bottleneck block, paired with squeeze-excitation attention."
        ),
    },
    "Sigmoid": {
        "title": "Sigmoid — Sigmoid Activation",
        "desc": (
            "The sigmoid function squashes any input into the range (0, 1) using the formula "
            "σ(x) = 1 / (1 + e^(-x)). In X3D-M, it's used inside the squeeze-excitation blocks to "
            "produce channel attention weights — values between 0 and 1 that indicate how important "
            "each feature channel is."
        ),
    },
    "AvgPool3d": {
        "title": "AvgPool3d — Average Pooling",
        "desc": (
            "Average pooling computes the mean value in a local window, reducing the spatial resolution. "
            "For example, a 2×2 pool over a 4×4 feature map produces a 2×2 output where each value is "
            "the average of a 2×2 patch. This downsamples the data, reducing computation in subsequent "
            "layers while retaining the overall structure of the features."
        ),
    },
    "AdaptiveAvgPool3d": {
        "title": "AdaptiveAvgPool3d — Global Average Pooling",
        "desc": (
            "Adaptive average pooling computes the mean across entire spatial and/or temporal dimensions "
            "to produce a fixed-size output regardless of input size. In the X3D-M head, it pools each "
            "channel down to a single value (1×1×1), collapsing the spatial and temporal axes into a "
            "compact feature vector used for classification."
        ),
    },
    "Linear": {
        "title": "Linear — Fully Connected Layer",
        "desc": (
            "A linear (fully connected) layer computes output = input × weights + bias. Every input "
            "feature is connected to every output feature. In X3D-M, the final linear layer maps the "
            "2048-dimensional feature vector to 400 class logits — one score per Kinetics-400 action class."
        ),
    },
    "Add": {
        "title": "Add — Residual (Skip) Connection",
        "desc": (
            "The Add operation implements the core idea of residual networks: it adds the output of a "
            "block's transformations back to the original input (the skip connection). This means each "
            "block only needs to learn the 'residual' difference, making very deep networks much easier "
            "to train and allowing gradients to flow directly through the skip path."
        ),
    },
    "Mul (element-wise)": {
        "title": "Mul — Element-wise Multiplication (SE Scaling)",
        "desc": (
            "Element-wise multiplication is used by the squeeze-excitation mechanism to scale each channel "
            "of the feature map by its learned importance weight. After SE computes a 0-to-1 attention "
            "score for each channel, Mul applies those scores — amplifying important channels and "
            "suppressing less useful ones."
        ),
    },
    "Dropout": {
        "title": "Dropout — Regularization",
        "desc": (
            "Dropout randomly sets a fraction of input values to zero during training to prevent "
            "overfitting. During inference (which is what these profiling runs measure), dropout is "
            "disabled — it simply passes data through unchanged. Its latency in these runs is effectively zero."
        ),
    },
    "Identity": {
        "title": "Identity — Pass-through",
        "desc": (
            "The Identity layer passes its input through unchanged. It's used as a placeholder in residual "
            "blocks where the skip connection doesn't need to change dimensions — the input already matches "
            "the output shape, so no transformation is needed."
        ),
    },
}

st.markdown(f"<p style='color:{TEXT_MUTED}; font-size:0.85rem; margin-bottom:0.6rem;'>Expand any category below to learn what it does.</p>", unsafe_allow_html=True)

# Group categories for display
conv_types = [n for n in cat_names if "conv" in n.lower()]
other_types = [n for n in cat_names if "conv" not in n.lower()]

for group_label, group_items in [("Convolution Layers", conv_types), ("Other Layers", other_types)]:
    if not group_items:
        continue
    st.markdown(f"<p style='color:{TEXT_PRIMARY}; font-weight:600; font-size:0.9rem; margin:0.8rem 0 0.3rem 0;'>{group_label}</p>", unsafe_allow_html=True)
    for name in group_items:
        info = CATEGORY_INFO.get(name, {"title": name, "desc": "No additional information available."})
        count = type_counts[name]
        with st.expander(f"{info['title']}  —  {count} layers"):
            st.markdown(f"<p style='color:{TEXT_MUTED}; font-size:0.88rem; line-height:1.65;'>{info['desc']}</p>", unsafe_allow_html=True)

st.markdown("<div style='height:1.6rem'></div>", unsafe_allow_html=True)

# ── Section 3 – Layer latency line chart ────────────────────────────────────
st.markdown(f"<h3 style='color:{TEXT_PRIMARY}; margin-bottom:0.5rem;'>Per-Layer Latency</h3>", unsafe_allow_html=True)

if not active_runs:
    st.info("Enable at least one run in the sidebar to view charts.")
else:
    fig_lines = go.Figure()
    for i, r in enumerate(active_runs):
        color = PALETTE[i % len(PALETTE)]
        latencies = [l["latency_ms"] for l in r["layers"]]
        layer_indices = list(range(1, len(latencies) + 1))
        fig_lines.add_trace(go.Scatter(
            x=layer_indices,
            y=latencies,
            mode="lines",
            name=r["_label"],
            line=dict(color=color, width=2),
            hovertemplate=(
                "<b>%{customdata[0]}</b><br>"
                "Layer #%{x}<br>"
                "Type: %{customdata[1]}<br>"
                "Latency: %{y:.2f} ms"
                "<extra>" + r["_label"] + "</extra>"
            ),
            customdata=[(l["name"], l["layer_type"]) for l in r["layers"]],
        ))

    fig_lines.update_layout(
        **PLOTLY_LAYOUT,
        height=460,
        margin=dict(l=60, r=30, t=50, b=50),
        xaxis=dict(title="Layer Index", gridcolor=BORDER, zerolinecolor=BORDER),
        yaxis=dict(title="Latency (ms)", gridcolor=BORDER, zerolinecolor=BORDER, type="log"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_lines)

st.markdown("<div style='height:1rem'></div>", unsafe_allow_html=True)

# ── Section 4 – Total latency bar chart ────────────────────────────────────
st.markdown(f"<h3 style='color:{TEXT_PRIMARY}; margin-bottom:0.5rem;'>Total Inference Latency</h3>", unsafe_allow_html=True)

if active_runs:
    labels = [r["_label"] for r in active_runs]
    totals = [r["total_latency_ms"] for r in active_runs]
    bar_colors = [PALETTE[i % len(PALETTE)] for i in range(len(active_runs))]

    # Format latency for display
    def fmt_latency(ms):
        if ms >= 1000:
            return f"{ms / 1000:.2f} s"
        return f"{ms:.1f} ms"

    fig_bar = go.Figure(go.Bar(
        x=labels,
        y=totals,
        marker_color=bar_colors,
        text=[fmt_latency(t) for t in totals],
        textposition="outside",
        textfont=dict(color=TEXT_PRIMARY, size=13),
    ))
    fig_bar.update_layout(
        **PLOTLY_LAYOUT,
        height=420,
        xaxis=dict(title="", gridcolor=BORDER, zerolinecolor=BORDER, tickangle=-20),
        yaxis=dict(title="Latency (ms)", gridcolor=BORDER, zerolinecolor=BORDER, type="log"),
        showlegend=False,
        margin=dict(l=60, r=30, t=50, b=100),
    )
    st.plotly_chart(fig_bar)

# ── Footer ──────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="text-align:center; padding:2rem 0 1rem 0; color:{TEXT_MUTED}; font-size:0.78rem;">
    X3D-M Scratch Library  ·  PyTorch-free Video Classification for RISC-V SoC
</div>
""", unsafe_allow_html=True)
