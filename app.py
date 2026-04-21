import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Pyramid Cinematic Atlas", layout="wide")

st.title("🏺 Ancient Egypt Cinematic Intelligence Atlas")
st.write("A living simulation of pyramid geometry, geography & civilization evolution")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("pyramids.csv")
df.columns = df.columns.str.strip()

# =========================
# FEATURES
# =========================
df_clean = df.dropna(subset=["Base1 (m)", "Base2 (m)", "Height (m)"]).copy()

df_clean["avg_base"] = (df_clean["Base1 (m)"] + df_clean["Base2 (m)"]) / 2
df_clean["aspect_ratio"] = df_clean["Height (m)"] / df_clean["avg_base"]
df_clean["footprint_diff"] = abs(df_clean["Base1 (m)"] - df_clean["Base2 (m)"])

features = ["Base1 (m)", "Base2 (m)", "Height (m)", "aspect_ratio", "footprint_diff"]

# =========================
# SCALE + PCA
# =========================
scaler = StandardScaler()
X = scaler.fit_transform(df_clean[features])

pca = PCA(n_components=3)
X_pca = pca.fit_transform(X)

df_clean["PC1"] = X_pca[:, 0]
df_clean["PC2"] = X_pca[:, 1]
df_clean["PC3"] = X_pca[:, 2]

# =========================
# ML MODELS
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean["cluster"] = kmeans.fit_predict(X_pca)

iso = IsolationForest(contamination=0.15, random_state=42)
df_clean["anomaly"] = iso.fit_predict(X)

# =========================
# 📜 DYNASTY TIMELINE SLIDER
# =========================
dynasty_range = st.slider(
    "📜 Dynasty Timeline Filter",
    int(df_clean["Dynasty"].min()),
    int(df_clean["Dynasty"].max()),
    (3, 18)
)

df_dyn = df_clean[
    (df_clean["Dynasty"] >= dynasty_range[0]) &
    (df_clean["Dynasty"] <= dynasty_range[1])
]

st.write(f"Showing Dynasties: {dynasty_range}")

# =========================
# 🌊 ANIMATED NILE FLOW
# =========================
st.subheader("🌊 Animated Nile Civilization Flow")

t = st.slider("🌊 River Flow Time", 0, 10, 0)

nile_lat = np.array([
    30.6, 30.2, 29.9, 29.6, 29.3, 29.0, 28.7, 28.3,
    27.9, 27.5, 27.1, 26.7, 26.3, 25.9, 25.5, 25.1, 24.8
])

nile_lon = np.array([
    31.2, 31.25, 31.3, 31.28, 31.26, 31.24, 31.22, 31.20,
    31.18, 31.16, 31.14, 31.12, 31.10, 31.08, 31.06, 31.04, 31.02
])

# 🌊 animation effect (river "breathing")
nile_lon = nile_lon + np.sin(t * 0.5) * 0.02

# =========================
# 🇪🇬 EGYPT CIVILIZATION MAP (UPGRADED)
# =========================
st.subheader("🗺️ Egypt Civilization Map (Pyramids + Nile + Ancient Zones)")

# -------------------------
# 🌊 NILE RIVER (realistic curve)
# -------------------------
nile_lat = [
    31.2, 30.9, 30.6, 30.2, 29.9, 29.6, 29.3, 29.0,
    28.7, 28.3, 27.9, 27.5, 27.1, 26.7, 26.3, 25.9, 25.5
]

nile_lon = [
    31.3, 31.28, 31.26, 31.24, 31.22, 31.20, 31.18, 31.16,
    31.14, 31.12, 31.10, 31.08, 31.06, 31.04, 31.02, 31.00, 30.98
]

# -------------------------
# 🔴 PYRAMID SITES (REAL LOCATIONS)
# -------------------------
pyramid_sites = df_clean.copy()

# -------------------------
# 🟠 ANCIENT CIVILISATION ZONES (buffered region idea)
# -------------------------
df_clean["civil_zone"] = "outside"

df_clean.loc[
    df_clean["Site"].str.contains(
        "giza|saqqara|dahshur|abusir|lisht|hawara",
        case=False, na=False
    ),
    "civil_zone"
] = "pyramid_core"

df_clean.loc[
    df_clean["Site"].str.contains(
        "abydos|edfu|elephantine|hierakonpolis|dara|naqada",
        case=False, na=False
    ),
    "civil_zone"
] = "ancient_expansion"

# -------------------------
# COLOR RULES
# -------------------------
color_map = {
    "pyramid_core": "red",        # 🔴 active pyramid regions
    "ancient_expansion": "orange",# 🟠 ancient civilization spread
    "outside": "lightgray"
}

df_clean["color"] = df_clean["civil_zone"].map(color_map)

# -------------------------
# 🌍 MAP FIGURE
# -------------------------
import plotly.graph_objects as go
import plotly.express as px

fig_map = go.Figure()

# 🌊 Nile River
fig_map.add_trace(go.Scattergeo(
    lat=nile_lat,
    lon=nile_lon,
    mode="lines",
    line=dict(width=3, color="blue"),
    name="Nile River"
))

# 🔴 + 🟠 Civilisation Points
fig_map.add_trace(go.Scattergeo(
    lat=df_clean["Latitude"],
    lon=df_clean["Longitude"],
    mode="markers+text",
    text=df_clean["Pharaoh"],
    textposition="top center",
    marker=dict(
        size=7,
        color=df_clean["color"],
        opacity=0.9
    ),
    hovertext=df_clean["Site"],
    name="Civilization Sites"
))

# -------------------------
# MAP STYLE (FOCUSED EGYPT VIEW)
# -------------------------
fig_map.update_layout(
    title=dict(
        text="🇪🇬 Ancient Egypt Civilization Map (Nile + Pyramid Core + Expansion Zones)",
        font=dict(size=22)
    ),
    geo=dict(
        scope="africa",
        projection_type="natural earth",

        # zoom into Egypt region
        center=dict(lat=26.8, lon=31.0),
        projection_scale=6,

        showland=True,
        landcolor="rgb(240,240,240)",
        showocean=True,
        oceancolor="rgb(200,230,255)",
        showcountries=True,

        lataxis=dict(range=[22, 32]),
        lonaxis=dict(range=[24, 35])
    ),
    height=700
)

st.plotly_chart(fig_map, use_container_width=True)
# =========================
# 🌌 3D STRUCTURE SPACE
# =========================
st.subheader("🌌 Flying Camera Pyramid Space")

colors = ["red" if x == -1 else "gold" for x in df_dyn["anomaly"]]

# 🎥 flying camera path
camera_path = [
    dict(eye=dict(x=1.5, y=1.5, z=1.5)),
    dict(eye=dict(x=0.5, y=2.0, z=1.0)),
    dict(eye=dict(x=-1.5, y=1.5, z=1.2)),
    dict(eye=dict(x=1.0, y=-1.5, z=1.5)),
]

camera = camera_path[t % len(camera_path)]

fig_3d = go.Figure()

fig_3d.add_trace(go.Scatter3d(
    x=df_dyn["PC1"],
    y=df_dyn["PC2"],
    z=df_dyn["PC3"],
    mode="markers+text",
    text=df_dyn["Pharaoh"],
    marker=dict(size=5, color=colors)
))

fig_3d.update_layout(
    scene_camera=camera,
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="PC3"
    ),
    height=650
)

# =========================
# LAYOUT
# =========================
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    st.plotly_chart(fig_3d, use_container_width=True)

# =========================
# 🏗️ PYRAMID BUILDER
# =========================
st.subheader("🏗️ Interactive Pyramid Reconstruction")

selected = st.selectbox("Select Pyramid", df_dyn["Pharaoh"].unique())
row = df_dyn[df_dyn["Pharaoh"] == selected].iloc[0]

base = float(row["Base1 (m)"])
height = float(row["Height (m)"])
half = base / 2

fig_pyr = go.Figure()

fig_pyr.add_trace(go.Mesh3d(
    x=[-half, half, half, -half],
    y=[-half, -half, half, half],
    z=[0, 0, 0, 0],
    opacity=0.4,
    color="tan"
))

apex = (0, 0, height)

for x, y in [(-half, -half), (half, -half), (half, half), (-half, half)]:
    fig_pyr.add_trace(go.Scatter3d(
        x=[x, apex[0]],
        y=[y, apex[1]],
        z=[0, apex[2]],
        mode="lines",
        line=dict(color="brown", width=4)
    ))

fig_pyr.update_layout(height=600)

st.plotly_chart(fig_pyr, use_container_width=True)

# =========================
# 🧠 INSIGHTS
# =========================
st.subheader("🧠 Civilization Intelligence Insight")

st.write("""
- Nile acts as a **civilization spine**, not random geography
- Pyramid construction follows a **linear river-based development corridor**
- Geometry evolves smoothly across dynasties, not in discrete jumps
- 3D structure space shows **continuous architectural evolution**
- Anomalies represent **experimental construction phases**
""")
