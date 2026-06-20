'''
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model
import io
import csv

from simulate import simulate_sequence
from optimized_path import compare_algorithms


# ══════════════════════════════════════════════
# 3-CLASS LABEL MAPPING  (matches training)
# ══════════════════════════════════════════════
DISEASE_LABELS = {
    0: "Pest Attack",
    1: "Overwatering",
    2: "Water Stress"
}

DISEASE_COLORS = {
    "Pest Attack":  "#ef4444",
    "Overwatering": "#3b82f6",
    "Water Stress": "#eab308",
}

DISCRETE_SCALE = [
    [0.00, "#ef4444"], [0.33, "#ef4444"],
    [0.33, "#3b82f6"], [0.67, "#3b82f6"],
    [0.67, "#eab308"], [1.00, "#eab308"],
]


# ══════════════════════════════════════════════
# CACHED MODEL LOAD
# ══════════════════════════════════════════════
@st.cache_resource
def _cached_model():
    root = Path(__file__).resolve().parent
    model_path = root / "crop_disease_convlstm.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load_model(str(model_path))


# ══════════════════════════════════════════════
# FARM ANALYSIS
# ══════════════════════════════════════════════
def run_farm_analysis(model, spray_percentile=90, seed=42):
    np.random.seed(seed)

    T, C = 12, 9
    PLOT_H, PLOT_W = 64, 64
    GRID_Y, GRID_X = 4, 4
    FARM_H = GRID_Y * PLOT_H
    FARM_W = GRID_X * PLOT_W

    farm_prob = np.zeros((FARM_H, FARM_W))
    farm_gt = np.zeros((FARM_H, FARM_W))
    plot_disease_id = np.zeros((GRID_Y, GRID_X), dtype=int)
    plot_confidence = np.zeros((GRID_Y, GRID_X))
    plot_simulated = np.empty((GRID_Y, GRID_X), dtype=object)

    for gy in range(GRID_Y):
        for gx in range(GRID_X):
            disease = np.random.choice(["pest", "overwater", "water_stress"])
            base = np.random.rand(T, PLOT_H, PLOT_W, C)

            seq, gt_masks = simulate_sequence(
                base,
                disease=disease,
                max_severity=np.random.uniform(0.3, 0.6)
            )

            noise = np.random.normal(0, 0.02, seq.shape)
            seq = np.clip(seq + noise, 0, 1)

            cls, msk = model.predict(np.expand_dims(seq, 0), verbose=0)

            disease_id = int(np.argmax(cls))
            confidence = float(np.max(cls))
            prob_mask = msk[0, ..., 0]

            y0, y1 = gy * PLOT_H, (gy+1) * PLOT_H
            x0, x1 = gx * PLOT_W, (gx+1) * PLOT_W

            farm_prob[y0:y1, x0:x1] = prob_mask
            farm_gt[y0:y1, x0:x1] = gt_masks[-1]
            plot_disease_id[gy, gx] = disease_id
            plot_confidence[gy, gx] = confidence
            plot_simulated[gy, gx] = disease

    norm = (farm_prob - farm_prob.min()) / (farm_prob.max() - farm_prob.min() + 1e-6)
    threshold = np.percentile(norm, spray_percentile)
    farm_spray = norm > threshold

    df, paths = compare_algorithms(farm_prob, farm_spray)
    best_algo = df.iloc[0]["Algorithm"]
    path = paths[best_algo]

    return {
        "farm_prob": farm_prob,
        "farm_spray": farm_spray,
        "path": path,
        "best_algo": best_algo,
        "grid_y": GRID_Y,
        "grid_x": GRID_X,
        "plot_disease_id": plot_disease_id,
        "plot_confidence": plot_confidence,
        "plot_simulated": plot_simulated,
    }


# ══════════════════════════════════════════════
# PLOTLY HELPERS
# ══════════════════════════════════════════════
def fig_route_map(farm_prob, farm_spray, path, show_spray):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=farm_prob, colorscale="Reds"))

    if show_spray and farm_spray.any():
        fig.add_trace(go.Heatmap(
            z=farm_spray.astype(np.float64),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,191,255,0.35)"]],
            showscale=False,
        ))

    if path:
        xs = [p[1] + 0.5 for p in path]
        ys = [p[0] + 0.5 for p in path]
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", line=dict(color="cyan")))
        fig.add_trace(go.Scatter(
            x=[xs[0]], y=[ys[0]],
            mode="markers",
            marker=dict(size=12, color="yellow", symbol="star")
        ))

    fig.update_layout(title="Farm Disease Map + Treatment Route", height=420)
    return fig


def fig_stress_grid(plot_disease_id, grid_y, grid_x):
    z = plot_disease_id.astype(float)
    text = [
        [DISEASE_LABELS[plot_disease_id[r, c]] for c in range(grid_x)]
        for r in range(grid_y)
    ]
    fig = go.Figure(data=go.Heatmap(
        z=z, text=text,
        texttemplate="%{text}",
        colorscale=DISCRETE_SCALE,
        showscale=False,
        zmin=0, zmax=2
    ))
    fig.update_layout(title="Predicted Stress per Plot", height=300)
    return fig


# ══════════════════════════════════════════════
# MAIN STREAMLIT APP
# ══════════════════════════════════════════════
def main():
    st.set_page_config(page_title="AgroScan", layout="wide", page_icon="🌾")

    # Header
    st.markdown("""
        <h1 style='font-family:Georgia,serif; color:#166534;'>
            🌾 AgroScan Farm Dashboard
        </h1>
        <p style='color:#6b7280; margin-top:-10px;'>
            Precision Agriculture · Disease Localization · Optimized Treatment
        </p>
        <hr style='border-color:#d1fae5;'>
    """, unsafe_allow_html=True)

    # Sidebar
    with st.sidebar:
        st.header("⚙️ Controls")
        seed = st.number_input("Random seed", min_value=0, value=42, step=1)
        spray_pct = st.slider("Spray zone percentile", 80, 98, 90)
        show_spray = st.checkbox("Highlight spray zones", True)
        show_sim = st.checkbox("Show simulated disease", True)
        run = st.button("▶ Run / Refresh Farm", use_container_width=True)

        st.markdown("---")
        st.caption("Disease key")
        for label, color in DISEASE_COLORS.items():
            st.markdown(
                f"<span style='color:{color}; font-size:18px;'>■</span> {label}",
                unsafe_allow_html=True
            )

    if "result" not in st.session_state:
        st.session_state.result = None

    if run:
        with st.spinner("Running farm analysis (this may take ~60 s)…"):
            model = _cached_model()
            result = run_farm_analysis(model, spray_pct, seed)
            st.session_state.result = result
        st.success("Analysis complete!")

    result = st.session_state.result
    if result is None:
        st.stop()

    farm_prob = result["farm_prob"]
    farm_spray = result["farm_spray"]
    path = result["path"]
    best_algo = result["best_algo"]
    gy, gx = result["grid_y"], result["grid_x"]
    pid = result["plot_disease_id"]
    conf = result["plot_confidence"]
    sim = result["plot_simulated"]

    # KPI + CSV
    k1, k2, k3, k4, k5 = st.columns([1,1,1,1,1])

    k1.metric("Total Plots", gy * gx)
    k2.metric("Spray Coverage", f"{100*farm_spray.mean():.1f}%")
    k3.metric("Treatment Steps", len(path))
    k4.metric("Best Algorithm", best_algo)

    with k5:
        st.markdown("### ⬇ CSV")
        if path:
            buf = io.StringIO()
            writer = csv.writer(buf)
            writer.writerow(["step", "row", "col"])
            for i, (y, x) in enumerate(path):
                writer.writerow([i, y, x])

            st.download_button(
                "Download",
                buf.getvalue().encode("utf-8"),
                "treatment_path.csv",
                "text/csv",
                use_container_width=True
            )
        else:
            st.caption("No path")

    st.markdown("---")

    # Charts
    col_map, col_grid = st.columns([3, 1])

    with col_map:
        st.plotly_chart(fig_route_map(farm_prob, farm_spray, path, show_spray), use_container_width=True)

    with col_grid:
        st.plotly_chart(fig_stress_grid(pid, gy, gx), use_container_width=True)

    st.markdown("---")

    # Table
    st.subheader("📋 Plot-by-Plot Report")

    rows = []
    for r in range(gy):
        for c in range(gx):
            row = {
                "Row": r,
                "Col": c,
                "Predicted Stress": DISEASE_LABELS[pid[r, c]],
                "Confidence %": round(float(conf[r, c]) * 100, 1),
            }
            if show_sim:
                row["Simulated Disease"] = str(sim[r, c])
                row["Match ✓/✗"] = (
                    "✓" if (
                        (sim[r, c] == "pest" and pid[r, c] == 0) or
                        (sim[r, c] == "overwater" and pid[r, c] == 1) or
                        (sim[r, c] == "water_stress" and pid[r, c] == 2)
                    ) else "✗"
                )
            rows.append(row)

    st.dataframe(rows, use_container_width=True)

    # Distribution
    st.subheader("📊 Disease Distribution")
    counts = {DISEASE_LABELS[k]: int(np.sum(pid == k)) for k in DISEASE_LABELS}
    col_a, col_b, col_c = st.columns(3)
    for col, (name, count) in zip([col_a, col_b, col_c], counts.items()):
        col.metric(name, count, f"{100*count/(gy*gx):.0f}% of plots")


if __name__ == "__main__":
    main()
'''
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from tensorflow.keras.models import load_model

from simulate import simulate_sequence
from optimized_path import compare_algorithms


# ══════════════════════════════════════════════
# 3-CLASS LABEL MAPPING  (matches training)
# ══════════════════════════════════════════════
DISEASE_LABELS = {
    0: "Pest Attack",
    1: "Overwatering",
    2: "Water Stress"
}

DISEASE_COLORS = {
    "Pest Attack":  "#ef4444",   # red
    "Overwatering": "#3b82f6",   # blue
    "Water Stress": "#eab308",   # yellow
}

DISCRETE_SCALE = [
    [0.00, "#ef4444"], [0.33, "#ef4444"],
    [0.33, "#3b82f6"], [0.67, "#3b82f6"],
    [0.67, "#eab308"], [1.00, "#eab308"],
]

# ══════════════════════════════════════════════
# GPS BOUNDING BOX  (Thanjavur farmland, Tamil Nadu)
# Covers the 256×256 pixel farm grid
# ══════════════════════════════════════════════
LAT_MAX = 10.7564   # top    (row = 0)
LAT_MIN = 10.7500   # bottom (row = FARM_H)
LON_MIN = 79.1300   # left   (col = 0)
LON_MAX = 79.1364   # right  (col = FARM_W)

PLOT_H, PLOT_W = 64, 64
GRID_Y, GRID_X = 4, 4
FARM_H         = GRID_Y * PLOT_H   # 256
FARM_W         = GRID_X * PLOT_W   # 256


# ══════════════════════════════════════════════
# GPS HELPERS
# ══════════════════════════════════════════════
def pixel_to_gps(row, col):
    """Convert pixel (row, col) → (latitude, longitude)."""
    lat = LAT_MAX - (row / FARM_H) * (LAT_MAX - LAT_MIN)
    lon = LON_MIN + (col / FARM_W) * (LON_MAX - LON_MIN)
    return round(lat, 6), round(lon, 6)


def build_waypoints_df(path, farm_prob, farm_spray, plot_disease_id):
    """
    Build a DataFrame of SPRAY-ONLY waypoints with GPS coordinates.

    Columns: Step, Pixel_Row, Pixel_Col, Latitude, Longitude,
             Severity, Spray, Disease_Type
    """
    rows = []
    step = 1
    for r, c in path:
        if not farm_spray[r, c]:
            continue                          # skip non-spray points
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


# ══════════════════════════════════════════════
# CACHED MODEL LOAD
# ══════════════════════════════════════════════
@st.cache_resource
def _cached_model():
    root       = Path(__file__).resolve().parent
    model_path = root / "crop_disease_convlstm.keras"
    if not model_path.is_file():
        raise FileNotFoundError(f"Model not found: {model_path}")
    return load_model(str(model_path))


# ══════════════════════════════════════════════
# FARM ANALYSIS
# ══════════════════════════════════════════════
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
                base,
                disease=disease,
                max_severity=np.random.uniform(0.3, 0.6)
            )

            noise = np.random.normal(0, 0.02, seq.shape)
            seq   = np.clip(seq + noise, 0, 1)

            cls, msk   = model.predict(np.expand_dims(seq, 0), verbose=0)
            disease_id = int(np.argmax(cls))
            confidence = float(np.max(cls))

            print("Prediction:", cls)

            prob_mask = msk[0, ..., 0]

            y0, y1 = gy * PLOT_H, (gy + 1) * PLOT_H
            x0, x1 = gx * PLOT_W, (gx + 1) * PLOT_W

            farm_prob[y0:y1, x0:x1] = prob_mask
            farm_gt[y0:y1, x0:x1]   = gt_masks[-1]
            plot_disease_id[gy, gx] = disease_id
            plot_confidence[gy, gx] = confidence
            plot_simulated[gy, gx]  = disease

    # Spray zone
    norm       = (farm_prob - farm_prob.min()) / (farm_prob.max() - farm_prob.min() + 1e-6)
    threshold  = np.percentile(norm, spray_percentile)
    farm_spray = norm > threshold

    # Best path
    df_algo, paths = compare_algorithms(farm_prob, farm_spray)
    best_algo      = df_algo.iloc[0]["Algorithm"]
    path           = paths[best_algo]

    # Spray-only waypoints with GPS
    df_waypoints = build_waypoints_df(path, farm_prob, farm_spray, plot_disease_id)

    return {
        "farm_prob":       farm_prob,
        "farm_spray":      farm_spray,
        "path":            path,
        "best_algo":       best_algo,
        "df_waypoints":    df_waypoints,
        "grid_y":          GRID_Y,
        "grid_x":          GRID_X,
        "plot_disease_id": plot_disease_id,
        "plot_confidence": plot_confidence,
        "plot_simulated":  plot_simulated,
    }


# ══════════════════════════════════════════════
# PLOTLY HELPERS
# ══════════════════════════════════════════════
def fig_route_map(farm_prob, farm_spray, path, show_spray):
    fig = go.Figure()
    fig.add_trace(go.Heatmap(z=farm_prob, colorscale="Reds", name="Disease Prob"))

    if show_spray and farm_spray.any():
        fig.add_trace(go.Heatmap(
            z=farm_spray.astype(np.float64),
            colorscale=[[0, "rgba(0,0,0,0)"], [1, "rgba(0,191,255,0.35)"]],
            showscale=False,
            name="Spray Zone"
        ))

    if path:
        xs = [p[1] + 0.5 for p in path]
        ys = [p[0] + 0.5 for p in path]
        fig.add_trace(go.Scatter(
            x=xs, y=ys, mode="lines",
            line=dict(color="cyan", width=2),
            name="Treatment Path"
        ))
        fig.add_trace(go.Scatter(
            x=[xs[0]], y=[ys[0]], mode="markers",
            marker=dict(size=12, color="yellow", symbol="star"),
            name="Start"
        ))

    fig.update_layout(
        title="Farm Disease Map + Treatment Route",
        height=420,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


def fig_stress_grid(plot_disease_id, grid_y, grid_x):
    z    = plot_disease_id.astype(float)
    text = [
        [DISEASE_LABELS[plot_disease_id[r, c]] for c in range(grid_x)]
        for r in range(grid_y)
    ]
    fig = go.Figure(data=go.Heatmap(
        z=z, text=text,
        texttemplate="%{text}",
        colorscale=DISCRETE_SCALE,
        showscale=False,
        zmin=0, zmax=2
    ))
    fig.update_layout(
        title="Predicted Stress per Plot",
        height=300,
        margin=dict(l=0, r=0, t=40, b=0)
    )
    return fig


# ══════════════════════════════════════════════
# MAIN STREAMLIT APP
# ══════════════════════════════════════════════
def main():
    st.set_page_config(page_title="AgroScan", layout="wide", page_icon="🌾")

    # ── Header ───────────────────────────────
    st.markdown("""
        <h1 style='font-family:Georgia,serif; color:#166534;'>
            🌾 AgroScan Farm Dashboard
        </h1>
        <p style='color:#6b7280; margin-top:-10px;'>
            Precision Agriculture · Disease Localization · Optimized Treatment
        </p>
        <hr style='border-color:#d1fae5;'>
    """, unsafe_allow_html=True)

    # ── Sidebar ───────────────────────────────
    with st.sidebar:
        st.header("⚙️ Controls")
        seed       = st.number_input("Random seed", min_value=0, value=42, step=1)
        spray_pct  = st.slider("Spray zone percentile", 80, 98, 90)
        show_spray = st.checkbox("Highlight spray zones", True)
        show_sim   = st.checkbox("Show simulated disease", True)
        run        = st.button("▶ Run / Refresh Farm", use_container_width=True)

        st.markdown("---")
        st.caption("Disease key")
        for label, color in DISEASE_COLORS.items():
            st.markdown(
                f"<span style='color:{color}; font-size:18px;'>■</span> {label}",
                unsafe_allow_html=True
            )

        st.markdown("---")
        # st.caption("📍 GPS Region: Thanjavur, Tamil Nadu")
        # st.markdown(f"""
        #     <small>
        #     Lat: {LAT_MIN} → {LAT_MAX}<br>
        #     Lon: {LON_MIN} → {LON_MAX}
        #     </small>
        # """, unsafe_allow_html=True)

    # ── Session state ─────────────────────────
    if "result" not in st.session_state:
        st.session_state.result = None

    if run:
        with st.spinner("Running farm analysis (this may take ~60 s)…"):
            model  = _cached_model()
            result = run_farm_analysis(model, spray_pct, seed)
            st.session_state.result = result
        st.success("Analysis complete!")

    result = st.session_state.result
    if result is None:
        st.info("👈 Press **Run / Refresh Farm** in the sidebar to start.")
        st.stop()

    farm_prob    = result["farm_prob"]
    farm_spray   = result["farm_spray"]
    path         = result["path"]
    best_algo    = result["best_algo"]
    df_waypoints = result["df_waypoints"]
    gy, gx       = result["grid_y"], result["grid_x"]
    pid          = result["plot_disease_id"]
    conf         = result["plot_confidence"]
    sim          = result["plot_simulated"]

    # ── KPI row ───────────────────────────────
    k1, k2, k3, k4, k5 = st.columns(5)
    k1.metric("Total Plots",     gy * gx)
    k2.metric("Spray Coverage",  f"{100 * farm_spray.mean():.1f}%")
    k3.metric("Treatment Steps", len(path))
    k4.metric("Best Algorithm",  best_algo)
    k5.metric("Spray Waypoints", len(df_waypoints))

    st.markdown("---")

    # ── Main charts ───────────────────────────
    col_map, col_grid = st.columns([3, 1])

    with col_map:
        st.plotly_chart(
            fig_route_map(farm_prob, farm_spray, path, show_spray),
            use_container_width=True
        )

    with col_grid:
        st.plotly_chart(
            fig_stress_grid(pid, gy, gx),
            use_container_width=True
        )

    st.markdown("---")

    # ── Spray Waypoints + CSV Download ───────
    st.subheader("📍 Spray Waypoints – GPS Coordinates")
    st.caption(
        f"Showing {len(df_waypoints)} spray-zone waypoints only. "
        "Download as CSV for drone / robot autonomous navigation."
    )

    st.dataframe(df_waypoints, use_container_width=True)

    csv_bytes = df_waypoints.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="⬇️ Download Spray Waypoints CSV",
        data=csv_bytes,
        file_name=f"agroscan_spray_waypoints_seed{seed}.csv",
        mime="text/csv",
        help="GPS coordinates of spray zones for drone/robot navigation"
    )

    st.markdown("---")

    # ── Per-plot table ────────────────────────
    st.subheader("📋 Plot-by-Plot Report")

    rows = []
    for r in range(gy):
        for c in range(gx):
            centre_row = r * PLOT_H + PLOT_H // 2
            centre_col = c * PLOT_W + PLOT_W // 2
            lat, lon   = pixel_to_gps(centre_row, centre_col)

            row = {
                "Row":              r,
                "Col":              c,
                "Predicted Stress": DISEASE_LABELS[pid[r, c]],
                "Confidence %":     round(float(conf[r, c]) * 100, 1),
                "Centre Lat":       lat,
                "Centre Lon":       lon,
            }
            if show_sim:
                row["Simulated Disease"] = str(sim[r, c])
                row["Match ✓/✗"] = (
                    "✓" if (
                        (sim[r, c] == "pest"         and pid[r, c] == 0) or
                        (sim[r, c] == "overwater"    and pid[r, c] == 1) or
                        (sim[r, c] == "water_stress" and pid[r, c] == 2)
                    ) else "✗"
                )
            rows.append(row)

    st.dataframe(rows, use_container_width=True)

    # ── Disease count summary ─────────────────
    st.subheader("📊 Disease Distribution")
    counts = {DISEASE_LABELS[k]: int(np.sum(pid == k)) for k in DISEASE_LABELS}
    col_a, col_b, col_c = st.columns(3)
    for col, (name, count) in zip([col_a, col_b, col_c], counts.items()):
        col.metric(name, count, f"{100 * count / (gy * gx):.0f}% of plots")


if __name__ == "__main__":
    main()