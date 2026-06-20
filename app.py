from pathlib import Path
import time

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from tensorflow.keras.models import load_model

from simulate import simulate_sequence
from optimized_path import (
    run_astar, run_mst, run_tsp, run_dijkstra, run_sa,
    path_distance, disease_gain, count_turns,
)


# ══════════════════════════════════════════════════════════════════
# CONSTANTS
# ══════════════════════════════════════════════════════════════════
DISEASE_LABELS = {0: "Pest Attack", 1: "Overwatering", 2: "Water Stress"}

DISEASE_COLORS = {
    "Pest Attack":  "#ef4444",
    "Overwatering": "#3b82f6",
    "Water Stress": "#eab308",
}

DISEASE_ICONS = {
    "Pest Attack":  "🐛",
    "Overwatering": "💧",
    "Water Stress": "🌵",
}

DISCRETE_SCALE = [
    [0.00, "#ef4444"], [0.33, "#ef4444"],
    [0.33, "#3b82f6"], [0.67, "#3b82f6"],
    [0.67, "#eab308"], [1.00, "#eab308"],
]

ALGO_COLORS = {
    "Direct A*":           "#06b6d4",
    "MST + A*":            "#8b5cf6",
    "TSP":                 "#f97316",
    "Dijkstra":            "#10b981",
    "Simulated Annealing": "#f43f5e",
}

LAT_MAX, LAT_MIN = 10.7564, 10.7500
LON_MIN, LON_MAX = 79.1300, 79.1364

PLOT_H, PLOT_W = 64, 64
GRID_Y, GRID_X = 4, 4
FARM_H = GRID_Y * PLOT_H   # 256
FARM_W = GRID_X * PLOT_W   # 256


# ══════════════════════════════════════════════════════════════════
# GPS HELPERS
# ══════════════════════════════════════════════════════════════════
def pixel_to_gps(row, col):
    lat = LAT_MAX - (row / FARM_H) * (LAT_MAX - LAT_MIN)
    lon = LON_MIN + (col / FARM_W) * (LON_MAX - LON_MIN)
    return round(lat, 6), round(lon, 6)


def build_waypoints_df(path, farm_prob, farm_spray, plot_disease_id):
    rows = []
    step = 1
    for r, c in path:
        if not farm_spray[r, c]:
            continue
        lat, lon = pixel_to_gps(r, c)
        gy = min(r // PLOT_H, GRID_Y - 1)
        gx = min(c // PLOT_W, GRID_X - 1)
        rows.append({
            "Step":         step,
            "Pixel_Row":    r,
            "Pixel_Col":    c,
            "Latitude":     lat,
            "Longitude":    lon,
            "Severity":     round(float(farm_prob[r, c]), 4),
            "Spray":        "Yes",
            "Disease_Type": DISEASE_LABELS[plot_disease_id[gy, gx]],
        })
        step += 1
    return pd.DataFrame(rows)


# ══════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════
@st.cache_resource
def _cached_model():
    root = Path(__file__).resolve().parent
    model_path = root / "crop_disease_convlstm.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load_model(str(model_path))


# ══════════════════════════════════════════════════════════════════
# ALGORITHM COMPARISON  (no plt.show — safe for Streamlit)
# ══════════════════════════════════════════════════════════════════
def _compare_algorithms(prob_map, spray_mask, start=(0, 0)):
    algos = {
        "Direct A*":           run_astar,
        "MST + A*":            run_mst,
        "TSP":                 run_tsp,
        "Dijkstra":            run_dijkstra,
        "Simulated Annealing": run_sa,
    }
    results, paths = [], {}
    for name, func in algos.items():
        t0   = time.time()
        path = func(prob_map, spray_mask, start)
        rt   = time.time() - t0
        dist = path_distance(path)
        gain = disease_gain(path, prob_map)
        results.append({
            "Algorithm":   name,
            "Distance":    round(dist, 2),
            "Gain":        round(gain, 2),
            "Efficiency":  round(gain / (dist + 1e-6), 4),
            "Turns":       count_turns(path),
            "Runtime (s)": round(rt, 4),
        })
        paths[name] = path
    df = pd.DataFrame(results).sort_values("Efficiency", ascending=False)
    return df, paths


# ══════════════════════════════════════════════════════════════════
# FARM ANALYSIS
# ══════════════════════════════════════════════════════════════════
def run_farm_analysis(model, spray_percentile=90, seed=42):
    np.random.seed(seed)
    T, C = 12, 9

    farm_prob       = np.zeros((FARM_H, FARM_W))
    farm_gt         = np.zeros((FARM_H, FARM_W))
    plot_disease_id = np.zeros((GRID_Y, GRID_X), dtype=int)
    plot_confidence = np.zeros((GRID_Y, GRID_X))
    plot_simulated  = np.empty((GRID_Y, GRID_X), dtype=object)

    for gy in range(GRID_Y):
        for gx in range(GRID_X):
            disease = np.random.choice(["pest", "overwater", "water_stress"])
            base    = np.random.rand(T, PLOT_H, PLOT_W, C)

            seq, gt_masks = simulate_sequence(
                base, disease=disease,
                max_severity=np.random.uniform(0.3, 0.6)
            )
            noise = np.random.normal(0, 0.02, seq.shape)
            seq   = np.clip(seq + noise, 0, 1)

            cls, msk   = model.predict(np.expand_dims(seq, 0), verbose=0)
            disease_id = int(np.argmax(cls))
            confidence = float(np.max(cls))
            prob_mask  = msk[0, ..., 0]

            y0, y1 = gy * PLOT_H, (gy + 1) * PLOT_H
            x0, x1 = gx * PLOT_W, (gx + 1) * PLOT_W
            farm_prob[y0:y1, x0:x1] = prob_mask
            farm_gt[y0:y1, x0:x1]   = gt_masks[-1]
            plot_disease_id[gy, gx] = disease_id
            plot_confidence[gy, gx] = confidence
            plot_simulated[gy, gx]  = disease

    norm       = (farm_prob - farm_prob.min()) / (farm_prob.max() - farm_prob.min() + 1e-6)
    threshold  = np.percentile(norm, spray_percentile)
    farm_spray = norm > threshold

    df_algo, all_paths = _compare_algorithms(farm_prob, farm_spray)
    best_algo  = df_algo.iloc[0]["Algorithm"]
    best_path  = all_paths[best_algo]

    df_waypoints = build_waypoints_df(best_path, farm_prob, farm_spray, plot_disease_id)

    return {
        "farm_prob":       farm_prob,
        "farm_spray":      farm_spray,
        "path":            best_path,
        "best_algo":       best_algo,
        "df_algo":         df_algo,
        "all_paths":       all_paths,
        "df_waypoints":    df_waypoints,
        "plot_disease_id": plot_disease_id,
        "plot_confidence": plot_confidence,
        "plot_simulated":  plot_simulated,
    }


# ══════════════════════════════════════════════════════════════════
# PLOTLY FIGURES
# ══════════════════════════════════════════════════════════════════
def fig_farm_map(farm_prob, farm_spray, path, show_spray):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(
        z=farm_prob, colorscale="Reds",
        colorbar=dict(title="Severity", thickness=12, len=0.8),
        name="Disease Prob"
    ))
    if show_spray and farm_spray.any():
        fig.add_trace(go.Heatmap(
            z=farm_spray.astype(float),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,191,255,0.4)"]],
            showscale=False, name="Spray Zone"
        ))
    if path:
        xs = [p[1] + 0.5 for p in path]
        ys = [p[0] + 0.5 for p in path]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="cyan", width=2), name="Treatment Route"
        ))
        fig.add_trace(go.Scatter(
            x=[xs[0]], y=[ys[0]], mode="markers",
            marker=dict(size=14, color="yellow", symbol="star"), name="Start"
        ))
        fig.add_trace(go.Scatter(
            x=[xs[-1]], y=[ys[-1]], mode="markers",
            marker=dict(size=11, color="lime", symbol="square"), name="End"
        ))
    for i in range(1, GRID_Y):
        fig.add_hline(y=i * PLOT_H, line=dict(color="white", width=1, dash="dot"))
    for i in range(1, GRID_X):
        fig.add_vline(x=i * PLOT_W, line=dict(color="white", width=1, dash="dot"))
    fig.update_layout(
        title="Farm Disease Heatmap + Optimised Treatment Route",
        height=440, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=-0.12, font=dict(size=11)),
    )
    return fig


def fig_stress_grid(plot_disease_id):
    text = [
        [DISEASE_LABELS[plot_disease_id[r, c]] for c in range(GRID_X)]
        for r in range(GRID_Y)
    ]
    fig = go.Figure(go.Heatmap(
        z=plot_disease_id.astype(float), text=text,
        texttemplate="%{text}", colorscale=DISCRETE_SCALE,
        showscale=False, zmin=0, zmax=2
    ))
    fig.update_layout(
        title="Per-Plot Stress Classification",
        height=300, margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def fig_gps_map(df_waypoints, path, plot_disease_id):
    fig = go.Figure()

    # Farm boundary
    fig.add_trace(go.Scattermapbox(
        lat=[LAT_MIN, LAT_MAX, LAT_MAX, LAT_MIN, LAT_MIN],
        lon=[LON_MIN, LON_MIN, LON_MAX, LON_MAX, LON_MIN],
        mode="lines", line=dict(color="white", width=2),
        name="Farm Boundary", hoverinfo="skip"
    ))

    # 4×4 grid lines
    for i in range(1, GRID_Y):
        lat = LAT_MAX - i * (LAT_MAX - LAT_MIN) / GRID_Y
        fig.add_trace(go.Scattermapbox(
            lat=[lat, lat], lon=[LON_MIN, LON_MAX], mode="lines",
            line=dict(color="rgba(255,255,255,0.35)", width=1),
            showlegend=False, hoverinfo="skip"
        ))
    for i in range(1, GRID_X):
        lon = LON_MIN + i * (LON_MAX - LON_MIN) / GRID_X
        fig.add_trace(go.Scattermapbox(
            lat=[LAT_MIN, LAT_MAX], lon=[lon, lon], mode="lines",
            line=dict(color="rgba(255,255,255,0.35)", width=1),
            showlegend=False, hoverinfo="skip"
        ))

    # Plot centre markers (one per 4×4 cell)
    lats, lons, colors, texts = [], [], [], []
    for gy in range(GRID_Y):
        for gx in range(GRID_X):
            cr = gy * PLOT_H + PLOT_H // 2
            cc = gx * PLOT_W + PLOT_W // 2
            lat, lon = pixel_to_gps(cr, cc)
            label = DISEASE_LABELS[plot_disease_id[gy, gx]]
            lats.append(lat)
            lons.append(lon)
            colors.append(DISEASE_COLORS[label])
            texts.append(f"Plot ({gy},{gx}): {DISEASE_ICONS[label]} {label}")
    fig.add_trace(go.Scattermapbox(
        lat=lats, lon=lons, mode="markers",
        marker=dict(size=16, color=colors, opacity=0.9),
        text=texts, hoverinfo="text", name="Plot Centres"
    ))

    # Spray waypoints grouped by disease type
    if not df_waypoints.empty:
        for dtype in df_waypoints["Disease_Type"].unique():
            sub = df_waypoints[df_waypoints["Disease_Type"] == dtype]
            fig.add_trace(go.Scattermapbox(
                lat=sub["Latitude"].tolist(),
                lon=sub["Longitude"].tolist(),
                mode="markers",
                marker=dict(size=6, color=DISEASE_COLORS.get(dtype, "#94a3b8"), opacity=0.65),
                name=f"Spray · {dtype}",
                hovertemplate="Step %{customdata[0]}<br>Severity: %{customdata[1]:.3f}",
                customdata=sub[["Step", "Severity"]].values,
            ))

    # Treatment route (sampled every 4th point for map performance)
    if path:
        route_lats, route_lons = [], []
        for r, c in path[::4]:
            lat, lon = pixel_to_gps(r, c)
            route_lats.append(lat)
            route_lons.append(lon)
        fig.add_trace(go.Scattermapbox(
            lat=route_lats, lon=route_lons, mode="lines",
            line=dict(color="cyan", width=2), name="Treatment Route", opacity=0.85
        ))
        s_lat, s_lon = pixel_to_gps(path[0][0], path[0][1])
        fig.add_trace(go.Scattermapbox(
            lat=[s_lat], lon=[s_lon], mode="markers",
            marker=dict(size=14, color="yellow"), name="Depot / Start"
        ))

    fig.update_layout(
        mapbox=dict(
            style="open-street-map",
            center=dict(lat=(LAT_MIN + LAT_MAX) / 2, lon=(LON_MIN + LON_MAX) / 2),
            zoom=16
        ),
        height=520, margin=dict(l=0, r=0, t=0, b=0),
        legend=dict(
            bgcolor="rgba(255,255,255,0.85)", bordercolor="#d1d5db",
            borderwidth=1, font=dict(size=11)
        )
    )
    return fig


def fig_algo_paths(farm_prob, all_paths, best_algo):
    names = list(all_paths.keys())
    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=[
            f"{'🏆 ' if n == best_algo else ''}{n}" for n in names
        ] + [""],
        vertical_spacing=0.1, horizontal_spacing=0.04
    )
    positions = [(1, 1), (1, 2), (1, 3), (2, 1), (2, 2)]
    for idx, name in enumerate(names):
        path = all_paths[name]
        r, c = positions[idx]
        fig.add_trace(go.Heatmap(
            z=farm_prob, colorscale="Reds", showscale=False
        ), row=r, col=c)
        if path:
            xs = [p[1] + 0.5 for p in path]
            ys = [p[0] + 0.5 for p in path]
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="lines",
                line=dict(
                    color=ALGO_COLORS.get(name, "cyan"),
                    width=3 if name == best_algo else 1.5
                ),
                showlegend=False
            ), row=r, col=c)
            fig.add_trace(go.Scatter(
                x=[xs[0]], y=[ys[0]], mode="markers",
                marker=dict(size=9, color="yellow", symbol="star"),
                showlegend=False
            ), row=r, col=c)
    fig.update_layout(
        height=580, showlegend=False,
        title=dict(text="All 5 Algorithm Paths — Best is Highlighted 🏆", font=dict(size=14)),
        margin=dict(l=0, r=0, t=55, b=0)
    )
    return fig


def fig_efficiency_bar(df_algo, best_algo):
    df = df_algo.sort_values("Efficiency", ascending=True)
    colors = [
        "#16a34a" if a == best_algo else ALGO_COLORS.get(a, "#94a3b8")
        for a in df["Algorithm"]
    ]
    fig = go.Figure(go.Bar(
        x=df["Efficiency"], y=df["Algorithm"], orientation="h",
        marker_color=colors,
        text=[f"{v:.4f}" for v in df["Efficiency"]], textposition="outside"
    ))
    fig.update_layout(
        title="Efficiency Score per Algorithm (Disease Gain ÷ Distance)",
        xaxis_title="Efficiency", height=280,
        margin=dict(l=0, r=60, t=40, b=0)
    )
    return fig


def fig_disease_pie(plot_disease_id):
    counts = {DISEASE_LABELS[k]: int(np.sum(plot_disease_id == k)) for k in DISEASE_LABELS}
    fig = go.Figure(go.Pie(
        labels=list(counts.keys()),
        values=list(counts.values()),
        marker_colors=[DISEASE_COLORS[k] for k in counts],
        hole=0.45, textinfo="label+percent+value"
    ))
    fig.update_layout(
        title="Disease Distribution Across Farm",
        height=300, margin=dict(l=0, r=0, t=40, b=0),
        legend=dict(orientation="h", y=-0.15)
    )
    return fig


def fig_confidence_bar(plot_confidence, plot_disease_id):
    labels, means, stds, colors = [], [], [], []
    for did, label in DISEASE_LABELS.items():
        mask = plot_disease_id == did
        vals = plot_confidence[mask] * 100 if mask.any() else np.array([0.0])
        labels.append(f"{DISEASE_ICONS[label]} {label}")
        means.append(float(np.mean(vals)))
        stds.append(float(np.std(vals)))
        colors.append(DISEASE_COLORS[label])
    fig = go.Figure(go.Bar(
        x=labels, y=means,
        error_y=dict(type="data", array=stds, visible=True),
        marker_color=colors,
        text=[f"{m:.1f}%" for m in means], textposition="outside"
    ))
    fig.update_layout(
        title="Mean Model Confidence per Disease Class",
        yaxis_title="Confidence (%)", yaxis_range=[0, 110],
        height=280, margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# ══════════════════════════════════════════════════════════════════
# CSS + CARD HELPERS
# ══════════════════════════════════════════════════════════════════
def inject_css():
    st.markdown("""
    <style>
    .stApp { background-color: #f0fdf4; }

    .kpi-card {
        background: white; border-radius: 12px;
        padding: 0.9rem 1rem; text-align: center;
        box-shadow: 0 2px 8px rgba(0,0,0,0.07);
        border-top: 4px solid #16a34a;
    }
    .kpi-label {
        font-size: 0.68rem; color: #6b7280; font-weight: 700;
        text-transform: uppercase; letter-spacing: 0.07em;
    }
    .kpi-value { font-size: 1.7rem; font-weight: 800; color: #14532d; line-height: 1.15; }
    .kpi-sub   { font-size: 0.68rem; color: #9ca3af; }

    .section-hdr {
        font-size: 0.95rem; font-weight: 700; color: #14532d;
        border-left: 4px solid #16a34a; padding-left: 0.6rem;
        margin: 1.1rem 0 0.5rem;
    }

    .algo-card {
        background: white; border-radius: 10px;
        padding: 0.7rem; text-align: center;
        box-shadow: 0 1px 5px rgba(0,0,0,0.06);
    }

    .disease-pill {
        display: inline-block; border-radius: 20px;
        padding: 2px 10px; font-size: 0.75rem; font-weight: 600;
        margin: 2px;
    }

    section[data-testid="stSidebar"] > div:first-child {
        background: linear-gradient(180deg, #14532d 0%, #166534 100%);
    }

    section[data-testid="stSidebar"] label,
    section[data-testid="stSidebar"] label span,
    section[data-testid="stSidebar"] p,
    section[data-testid="stSidebar"] .stMarkdown p,
    section[data-testid="stSidebar"] .stSlider p,
    section[data-testid="stSidebar"] .stNumberInput label,
    section[data-testid="stSidebar"] .stCheckbox label,
    section[data-testid="stSidebar"] .stCheckbox span {
        color: white !important;
    }
    </style>
    """, unsafe_allow_html=True)


def kpi_card(label, value, sub=""):
    return (
        f"<div class='kpi-card'>"
        f"<div class='kpi-label'>{label}</div>"
        f"<div class='kpi-value'>{value}</div>"
        f"<div class='kpi-sub'>{sub}</div>"
        f"</div>"
    )


# ══════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════
def main():
    st.set_page_config(
        page_title="AgroScan", layout="wide",
        page_icon="🌾", initial_sidebar_state="expanded"
    )
    inject_css()

    # ── Header ─────────────────────────────────────────────────────
    st.markdown("""
        <div style="background:linear-gradient(135deg,#14532d,#15803d);
                    padding:1.4rem 2rem;border-radius:14px;margin-bottom:1.4rem;
                    box-shadow:0 4px 16px rgba(0,0,0,0.14);">
            <h1 style="color:white;margin:0;font-family:Georgia,serif;font-size:1.9rem;">
                🌾 AgroScan
            </h1>
            <p style="color:#bbf7d0;margin:3px 0 0;font-size:0.9rem;">
                Precision Agriculture · Disease Localisation · Optimised Drone Treatment
            </p>
            <div style="margin-top:0.55rem;font-size:0.78rem;">
                <span style="background:#166534;color:#bbf7d0;border-radius:20px;
                             padding:2px 10px;margin-right:6px;">
                    📍 Thanjavur, Tamil Nadu</span>
                <span style="background:#166534;color:#bbf7d0;border-radius:20px;
                             padding:2px 10px;margin-right:6px;">
                    4 × 4 Farm Grid · 256 × 256 px</span>
                <span style="background:#166534;color:#bbf7d0;border-radius:20px;
                             padding:2px 10px;">
                    ConvLSTM2D Dual-Head Model</span>
            </div>
        </div>
    """, unsafe_allow_html=True)

    # ── Sidebar ─────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("<h3 style='color:white;margin-top:0;'>⚙️ Controls</h3>",
                    unsafe_allow_html=True)
        seed      = st.number_input("🎲 Random Seed", min_value=0, value=42, step=1)
        spray_pct = st.slider(
            "💧 Spray Percentile", 80, 98, 90,
            help="Pixels above this severity percentile are marked as spray zones."
        )
        show_spray = st.checkbox("Highlight spray zones", True)
        show_sim   = st.checkbox("Show simulated disease", True)
        st.markdown("---")
        run = st.button("▶ Run Farm Analysis", use_container_width=True, type="primary")
        st.markdown("---")
        st.markdown("<p style='color:#86efac;font-size:0.78rem;font-weight:600;'>Disease Key</p>",
                    unsafe_allow_html=True)
        for label, color in DISEASE_COLORS.items():
            st.markdown(
                f"<span style='color:{color};font-size:15px;'>■</span> "
                f"<span style='color:white;font-size:0.82rem;'>"
                f"{DISEASE_ICONS[label]} {label}</span>",
                unsafe_allow_html=True
            )
        st.markdown("---")
        st.markdown(
            "<p style='color:#6ee7b7;font-size:0.72rem;line-height:1.7;'>"
            "📡 <b style='color:white;'>GPS Region</b><br>"
            "Lat: 10.7500 – 10.7564<br>"
            "Lon: 79.1300 – 79.1364</p>",
            unsafe_allow_html=True
        )

    # ── Session state ───────────────────────────────────────────────
    if "result" not in st.session_state:
        st.session_state.result = None

    if run:
        with st.spinner("🌾 Running farm analysis — this may take ~60 s…"):
            model  = _cached_model()
            result = run_farm_analysis(model, spray_pct, seed)
            st.session_state.result = result
        st.success("✅ Analysis complete! Explore the tabs below.")

    result = st.session_state.result

    # ── Welcome screen ──────────────────────────────────────────────
    if result is None:
        st.markdown("""
        <div style="background:white;border-radius:14px;padding:2.5rem 2rem;
                    box-shadow:0 2px 12px rgba(0,0,0,0.06);margin-top:1rem;text-align:center;">
            <h2 style="color:#14532d;margin-top:0;">How AgroScan Works</h2>
            <p style="color:#6b7280;">
                Press <b>Run Farm Analysis</b> in the sidebar to start.</p>
            <div style="display:flex;justify-content:center;gap:2.5rem;
                        margin-top:1.5rem;flex-wrap:wrap;">
                <div style="max-width:130px;">
                    <div style="font-size:2.8rem;">🛰️</div>
                    <b style="color:#14532d;font-size:0.9rem;">1. Simulate</b>
                    <p style="font-size:0.78rem;color:#6b7280;margin-top:4px;">
                        9-band Sentinel-2 data for 4×4 plots (T=12 timesteps)</p>
                </div>
                <div style="max-width:130px;">
                    <div style="font-size:2.8rem;">🤖</div>
                    <b style="color:#14532d;font-size:0.9rem;">2. Detect</b>
                    <p style="font-size:0.78rem;color:#6b7280;margin-top:4px;">
                        ConvLSTM2D classifies disease + generates probability mask</p>
                </div>
                <div style="max-width:130px;">
                    <div style="font-size:2.8rem;">🗺️</div>
                    <b style="color:#14532d;font-size:0.9rem;">3. Map</b>
                    <p style="font-size:0.78rem;color:#6b7280;margin-top:4px;">
                        Stitch 16 predictions → 256×256 farm disease heatmap</p>
                </div>
                <div style="max-width:130px;">
                    <div style="font-size:2.8rem;">🚁</div>
                    <b style="color:#14532d;font-size:0.9rem;">4. Route</b>
                    <p style="font-size:0.78rem;color:#6b7280;margin-top:4px;">
                        5 algorithms compete — best route drives autonomous treatment</p>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.stop()

    # ── Unpack ──────────────────────────────────────────────────────
    farm_prob    = result["farm_prob"]
    farm_spray   = result["farm_spray"]
    path         = result["path"]
    best_algo    = result["best_algo"]
    df_waypoints = result["df_waypoints"]
    df_algo      = result["df_algo"]
    all_paths    = result["all_paths"]
    pid          = result["plot_disease_id"]
    conf         = result["plot_confidence"]
    sim          = result["plot_simulated"]

    # ── KPI Row ─────────────────────────────────────────────────────
    kpi_data = [
        ("Total Plots",     f"{GRID_Y * GRID_X}",             "4 × 4 grid"),
        ("Spray Coverage",  f"{100*farm_spray.mean():.1f}%",  f"≥ {spray_pct}th %ile"),
        ("Treatment Steps", f"{len(path):,}",                  "pixel waypoints"),
        ("Best Algorithm",  best_algo,                         "by efficiency"),
        ("Spray Waypoints", f"{len(df_waypoints)}",            "GPS-tagged"),
        ("Avg Confidence",  f"{100*conf.mean():.1f}%",         "model certainty"),
    ]
    for col, (label, value, sub) in zip(st.columns(6), kpi_data):
        col.markdown(kpi_card(label, value, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Tabs ────────────────────────────────────────────────────────
    t1, t2, t3, t4, t5 = st.tabs([
        "🌾 Farm Overview",
        "🗺️  GPS Map",
        "📊 Analytics",
        "🤖 Algorithm Comparison",
        "📋 Reports",
    ])

    # ═══ Tab 1 — Farm Overview ══════════════════════════════════════
    with t1:
        col_map, col_side = st.columns([3, 1])
        with col_map:
            st.plotly_chart(
                fig_farm_map(farm_prob, farm_spray, path, show_spray),
                use_container_width=True
            )
        with col_side:
            st.plotly_chart(fig_stress_grid(pid), use_container_width=True)
            st.markdown("<div class='section-hdr'>Plot Summary</div>",
                        unsafe_allow_html=True)
            counts = {DISEASE_LABELS[k]: int(np.sum(pid == k)) for k in DISEASE_LABELS}
            for name, count in counts.items():
                color = DISEASE_COLORS[name]
                icon  = DISEASE_ICONS[name]
                st.markdown(
                    f"<div style='background:white;border-radius:8px;padding:5px 10px;"
                    f"margin:3px 0;border-left:3px solid {color};font-size:0.82rem;'>"
                    f"{icon} <b>{name}</b><br>"
                    f"<span style='color:#6b7280;'>{count} plot(s) · "
                    f"{100*count/(GRID_Y*GRID_X):.0f}%</span>"
                    f"</div>",
                    unsafe_allow_html=True
                )

    # ═══ Tab 2 — GPS Map ════════════════════════════════════════════
    with t2:
        st.markdown(
            "<div class='section-hdr'>📍 Real-world GPS Map — Thanjavur, Tamil Nadu</div>",
            unsafe_allow_html=True
        )
        st.caption(
            "Disease zone centres, spray waypoints, and treatment route plotted on "
            "OpenStreetMap satellite coordinates. Route sampled every 4 pixels for performance."
        )
        st.plotly_chart(fig_gps_map(df_waypoints, path, pid), use_container_width=True)
        c1, c2, c3 = st.columns(3)
        c1.metric("Spray Waypoints on Map", len(df_waypoints))
        c2.metric("Route Sampling", "Every 4 pixels")
        c3.metric("Coordinate System", "WGS-84")

    # ═══ Tab 3 — Analytics ══════════════════════════════════════════
    with t3:
        st.markdown("<div class='section-hdr'>Disease Distribution & Model Confidence</div>",
                    unsafe_allow_html=True)
        ca, cb = st.columns(2)
        with ca:
            st.plotly_chart(fig_disease_pie(pid), use_container_width=True)
        with cb:
            st.plotly_chart(fig_confidence_bar(conf, pid), use_container_width=True)

        st.markdown("<div class='section-hdr'>Algorithm Efficiency Ranking</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(fig_efficiency_bar(df_algo, best_algo), use_container_width=True)

    # ═══ Tab 4 — Algorithm Comparison ═══════════════════════════════
    with t4:
        st.markdown("<div class='section-hdr'>All 5 Paths — Side by Side</div>",
                    unsafe_allow_html=True)
        st.plotly_chart(
            fig_algo_paths(farm_prob, all_paths, best_algo),
            use_container_width=True
        )
        st.markdown("<div class='section-hdr'>Per-Algorithm Metrics</div>",
                    unsafe_allow_html=True)
        cols5 = st.columns(5)
        for col, (_, row) in zip(cols5, df_algo.iterrows()):
            name  = row["Algorithm"]
            color = "#16a34a" if name == best_algo else ALGO_COLORS.get(name, "#94a3b8")
            badge = "🏆 Best" if name == best_algo else ""
            col.markdown(
                f"<div class='algo-card' style='border-top:3px solid {color};'>"
                f"<div style='font-size:0.68rem;font-weight:700;color:#6b7280;'>{name}</div>"
                f"<div style='font-size:1.4rem;font-weight:800;color:{color};'>"
                f"{row['Efficiency']:.4f}</div>"
                f"<div style='font-size:0.68rem;color:#9ca3af;'>efficiency</div>"
                f"<div style='font-size:0.78rem;color:#374151;margin-top:5px;'>"
                f"📏 {row['Distance']:.0f} px<br>"
                f"↩️ {row['Turns']} turns<br>"
                f"⏱ {row['Runtime (s)']:.3f} s</div>"
                f"<div style='font-size:0.8rem;margin-top:4px;'>{badge}</div>"
                f"</div>",
                unsafe_allow_html=True
            )
        st.markdown("<br>", unsafe_allow_html=True)
        st.dataframe(df_algo, use_container_width=True, hide_index=True)

    # ═══ Tab 5 — Reports ════════════════════════════════════════════
    with t5:
        st.markdown("<div class='section-hdr'>📍 Spray Waypoints — GPS Coordinates</div>",
                    unsafe_allow_html=True)
        st.caption(
            f"{len(df_waypoints)} spray-zone waypoints · "
            "Download as CSV for drone / robot autonomous navigation."
        )
        st.dataframe(df_waypoints, use_container_width=True)
        st.download_button(
            "⬇️ Download Spray Waypoints CSV",
            data=df_waypoints.to_csv(index=False).encode("utf-8"),
            file_name=f"agroscan_spray_waypoints_seed{seed}.csv",
            mime="text/csv"
        )

        st.markdown("---")
        st.markdown("<div class='section-hdr'>📋 Plot-by-Plot Report</div>",
                    unsafe_allow_html=True)
        rows = []
        for r in range(GRID_Y):
            for c in range(GRID_X):
                cr, cc   = r * PLOT_H + PLOT_H // 2, c * PLOT_W + PLOT_W // 2
                lat, lon = pixel_to_gps(cr, cc)
                rd = {
                    "Plot":             f"({r},{c})",
                    "Predicted Stress": DISEASE_LABELS[pid[r, c]],
                    "Confidence %":     round(float(conf[r, c]) * 100, 1),
                    "Centre Lat":       lat,
                    "Centre Lon":       lon,
                }
                if show_sim:
                    rd["Simulated"]  = str(sim[r, c])
                    rd["Match ✓/✗"] = "✓" if (
                        (sim[r, c] == "pest"         and pid[r, c] == 0) or
                        (sim[r, c] == "overwater"    and pid[r, c] == 1) or
                        (sim[r, c] == "water_stress" and pid[r, c] == 2)
                    ) else "✗"
                rows.append(rd)
        st.dataframe(rows, use_container_width=True)

        st.markdown("---")
        st.markdown("<div class='section-hdr'>📊 Disease Distribution Summary</div>",
                    unsafe_allow_html=True)
        counts = {DISEASE_LABELS[k]: int(np.sum(pid == k)) for k in DISEASE_LABELS}
        for col, (name, count) in zip(st.columns(3), counts.items()):
            col.metric(
                f"{DISEASE_ICONS[name]} {name}",
                count,
                f"{100*count/(GRID_Y*GRID_X):.0f}% of plots"
            )


if __name__ == "__main__":
    main()
