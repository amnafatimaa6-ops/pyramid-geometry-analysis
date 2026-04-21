import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Pyramid Atlas", layout="wide")

st.title("🏺 Ancient Egypt Pyramid Intelligence Atlas")
st.write("A computational archaeology system mapping geometry + geography + anomalies")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("pyramids.csv")
df.columns = df.columns.str.strip()

# =========================
# CLEAN + FEATURE ENGINEERING
# =========================
df_clean = df.dropna(subset=["Base1 (m)", "Base2 (m)", "Height (m)"]).copy()

df_clean["avg_base"] = (df_clean["Base1 (m)"] + df_clean["Base2 (m)"]) / 2
df_clean["aspect_ratio"] = df_clean["Height (m)"] / df_clean["avg_base"]
df_clean["footprint_diff"] = abs(df_clean["Base1 (m)"] - df_clean["Base2 (m)"])

features = ["Base1 (m)", "Base2 (m)", "Height (m)", "aspect_ratio", "footprint_diff"]

# =========================
# SCALE
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# =========================
# PCA (3D STRUCTURE SPACE)
# =========================
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

df_clean["PC1"] = X_pca[:, 0]
df_clean["PC2"] = X_pca[:, 1]
df_clean["PC3"] = X_pca[:, 2]

# =========================
# CLUSTER + ANOMALY
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean["cluster"] = kmeans.fit_predict(X_pca)

iso = IsolationForest(contamination=0.15, random_state=42)
df_clean["anomaly"] = iso.fit_predict(X_scaled)

# =========================
# 🌍 CIVILIZATION ZONES (MAP LOGIC)
# =========================
df_clean["zone"] = "other"

df_clean.loc[
    df_clean["Site"].str.contains(
        "giza|saqqara|dahshur|abusir|lisht|hawara|el-lahun|south saqqara|north saqqara",
        case=False, na=False
    ),
    "zone"
] = "major_pyramid_zone"

df_clean.loc[
    df_clean["Site"].str.contains(
        "abydos|edfu|elephantine|hierakonpolis|dara",
        case=False, na=False
    ),
    "zone"
] = "ancient_upper_egypt_zone"

# COLORS
color_map = {
    "major_pyramid_zone": "gold",
    "ancient_upper_egypt_zone": "orange",
    "other": "lightblue"
}

df_clean["color"] = df_clean["zone"].map(color_map)

# =========================
# 🗺️ EGYPT MAP VIEW
# =========================
st.subheader("🗺️ Civilization & Pyramid Distribution Map")

fig_map = px.scatter_geo(
    df_clean,
    lat="Latitude",
    lon="Longitude",
    hover_name="Pharaoh",
    hover_data=["Site", "Base1 (m)", "Height (m)", "zone"],
    color="zone",
    color_discrete_map=color_map,
    projection="natural earth"
)

fig_map.update_layout(
    title=dict(
        text="🏺 Ancient Egypt Civilization Zones & Pyramid Sites",
        font=dict(size=28)
    ),
    geo=dict(
        showland=True,
        landcolor="rgb(235, 235, 235)",
        showcountries=True,
        showocean=True,
        oceancolor="rgb(210, 230, 255)",
        center=dict(lat=26.8, lon=31.0),
        projection_scale=4.5
    ),
    height=650
)

# =========================
# 🌌 3D STRUCTURAL SPACE
# =========================
colors = ["red" if x == -1 else "gold" for x in df_clean["anomaly"]]

fig_3d = go.Figure()

fig_3d.add_trace(go.Scatter3d(
    x=df_clean["PC1"],
    y=df_clean["PC2"],
    z=df_clean["PC3"],
    mode="markers+text",
    text=df_clean["Pharaoh"],
    textposition="top center",
    marker=dict(size=6, color=colors, opacity=0.85)
))

fig_3d.update_layout(
    title=dict(
        text="🏺 3D Pyramid Structural Intelligence Space",
        font=dict(size=30)
    ),
    scene=dict(
        xaxis_title="Structural Axis 1",
        yaxis_title="Structural Axis 2",
        zaxis_title="Structural Depth"
    ),
    height=600,
    margin=dict(l=0, r=0, b=0, t=80)
)

# =========================
# SIDE BY SIDE VIEW
# =========================
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    st.plotly_chart(fig_3d, use_container_width=True)

# =========================
# 🏗️ INDIVIDUAL PYRAMID VIEWER
# =========================
st.subheader("🏗️ Pyramid 3D Builder")

selected = st.selectbox("Select Pyramid", df_clean["Pharaoh"].unique())
row = df_clean[df_clean["Pharaoh"] == selected].iloc[0]

base = float(row["Base1 (m)"])
height = float(row["Height (m)"])
half = base / 2

fig_pyr = go.Figure()

fig_pyr.add_trace(go.Mesh3d(
    x=[-half, half, half, -half],
    y=[-half, -half, half, half],
    z=[0, 0, 0, 0],
    color="tan",
    opacity=0.5
))

apex = (0, 0, height)

base_points = [
    (-half, -half, 0),
    (half, -half, 0),
    (half, half, 0),
    (-half, half, 0)
]

for x, y, z in base_points:
    fig_pyr.add_trace(go.Scatter3d(
        x=[x, apex[0]],
        y=[y, apex[1]],
        z=[z, apex[2]],
        mode="lines",
        line=dict(color="brown", width=5)
    ))

fig_pyr.update_layout(
    title=dict(
        text=f"🏺 {selected} - 3D Structure",
        font=dict(size=26)
    ),
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Height"
    ),
    height=600,
    margin=dict(l=0, r=0, b=0, t=80)
)

st.plotly_chart(fig_pyr, use_container_width=True)

# =========================
# 🧠 INSIGHT PANEL
# =========================
st.subheader("🧠 Archaeological Intelligence Insight")

st.write("""
- Egypt shows strong **Nile-centered civilization clustering**
- Pyramid construction is highly concentrated in **Giza–Saqqara–Dahshur corridor**
- Structural geometry forms a **continuous evolutionary system, not separate categories**
- Anomalies represent **experimental architectural phases or transitional dynasties**
- Ancient Egypt behaves like a **stable engineering civilization with regional expansion nodes**
""")
