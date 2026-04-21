import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import time

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# =========================
# PAGE CONFIG (CINEMATIC STYLE)
# =========================
st.set_page_config(page_title="Pyramid Cinematic Atlas", layout="wide")

st.markdown(
    """
    <style>
    body {
        background-color: black;
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True
)

st.title("🎬 Ancient Egypt: Cinematic Pyramid Intelligence System")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("pyramids.csv")
df.columns = df.columns.str.strip()

# =========================
# FEATURE ENGINEERING
# =========================
df = df.dropna(subset=["Base1 (m)", "Base2 (m)", "Height (m)"])

df["avg_base"] = (df["Base1 (m)"] + df["Base2 (m)"]) / 2
df["aspect_ratio"] = df["Height (m)"] / df["avg_base"]

features = ["Base1 (m)", "Base2 (m)", "Height (m)", "aspect_ratio"]

X = StandardScaler().fit_transform(df[features])

# =========================
# PCA 3D
# =========================
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

df["PC1"] = X_pca[:, 0]
df["PC2"] = X_pca[:, 1]
df["PC3"] = X_pca[:, 2]

# =========================
# ANOMALY DETECTION
# =========================
iso = IsolationForest(contamination=0.12, random_state=42)
df["anomaly"] = iso.fit_predict(X)

# =========================
# 🌊 CINEMATIC NILE FLOW (ANIMATED LOOK)
# =========================
nile_lat = np.linspace(30.6, 24.8, 40)
nile_lon = np.linspace(31.25, 31.02, 40)

# =========================
# 🎬 SCENE 1 — INTRO TEXT
# =========================
st.markdown("## 🎞️ Scene 1: The Civilization Corridor")
st.write("""
> “Along a single river, one of history’s greatest engineering civilizations emerged.  
> This is not random geography — this is structured intelligence shaped by water, survival, and time.”
""")

# =========================
# 🌍 MAP (CINEMATIC GLOW STYLE)
# =========================
fig_map = go.Figure()

# pyramids glow
fig_map.add_trace(go.Scattergeo(
    lat=df["Latitude"],
    lon=df["Longitude"],
    mode="markers",
    marker=dict(
        size=7,
        color=df["anomaly"],
        colorscale="Hot",
        opacity=0.8
    ),
    text=df["Pharaoh"]
))

# nile river (animated illusion via gradient points)
fig_map.add_trace(go.Scattergeo(
    lat=nile_lat,
    lon=nile_lon,
    mode="lines",
    line=dict(width=3, color="cyan"),
    name="Nile River"
))

fig_map.update_layout(
    geo=dict(
        projection="natural earth",
        showland=True,
        landcolor="rgb(10,10,10)",
        oceancolor="rgb(0,30,60)",
        showocean=True,
        center=dict(lat=26.8, lon=31.0),
        projection_scale=4.8
    ),
    paper_bgcolor="black",
    height=600
)

st.plotly_chart(fig_map, use_container_width=True)

# =========================
# 🎬 SCENE 2 — 3D PYRAMID SPACE
# =========================
st.markdown("## 🎞️ Scene 2: Hidden Structural Dimension")

st.write("""
> “When geometry is compressed into higher dimensions, patterns emerge that are invisible in raw space.  
> This is where anomalies reveal themselves.”
""")

colors = ["red" if x == -1 else "gold" for x in df["anomaly"]]

fig_3d = go.Figure()

fig_3d.add_trace(go.Scatter3d(
    x=df["PC1"],
    y=df["PC2"],
    z=df["PC3"],
    mode="markers",
    marker=dict(size=5, color=colors, opacity=0.85)
))

fig_3d.update_layout(
    scene=dict(
        xaxis_title="Structural Axis 1",
        yaxis_title="Structural Axis 2",
        zaxis_title="Structural Depth"
    ),
    paper_bgcolor="black",
    height=650
)

st.plotly_chart(fig_3d, use_container_width=True)

# =========================
# 🎬 SCENE 3 — PYRAMID CLOSE-UP
# =========================
st.markdown("## 🎞️ Scene 3: Individual Monument Reconstruction")

selected = st.selectbox("Select Pyramid", df["Pharaoh"].unique())
row = df[df["Pharaoh"] == selected].iloc[0]

base = row["Base1 (m)"]
h = row["Height (m)"]
half = base / 2

fig_pyr = go.Figure()

# base
fig_pyr.add_trace(go.Mesh3d(
    x=[-half, half, half, -half],
    y=[-half, -half, half, half],
    z=[0, 0, 0, 0],
    color="tan",
    opacity=0.5
))

apex = (0, 0, h)

for x, y in [(-half,-half),(half,-half),(half,half),(-half,half)]:
    fig_pyr.add_trace(go.Scatter3d(
        x=[x, apex[0]],
        y=[y, apex[1]],
        z=[0, apex[2]],
        mode="lines",
        line=dict(color="white", width=3)
    ))

fig_pyr.update_layout(
    paper_bgcolor="black",
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    ),
    height=600
)

st.plotly_chart(fig_pyr, use_container_width=True)

# =========================
# 🎬 FINAL NARRATION
# =========================
st.markdown("## 🎬 Final Narration")

st.success("""
Across dynasties, Egypt does not behave like scattered construction —  
it behaves like a **designed system aligned along a river spine**.

Geometry is consistent.  
Anomalies are rare.  
Variation is evolutionary, not chaotic.

This is not just archaeology.

This is **structured civilization engineering over time**.
""")
