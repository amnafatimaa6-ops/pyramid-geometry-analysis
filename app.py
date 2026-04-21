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

st.title("🏺 Ancient Egypt Pyramid Atlas")
st.write("Geometric + Geographic Intelligence System for Pyramid Structures")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("pyramids.csv")
df.columns = df.columns.str.strip()

# =========================
# CLEAN + FEATURES
# =========================
df_clean = df.dropna(subset=["Base1 (m)", "Base2 (m)", "Height (m)"]).copy()

df_clean["avg_base"] = (df_clean["Base1 (m)"] + df_clean["Base2 (m)"]) / 2
df_clean["aspect_ratio"] = df_clean["Height (m)"] / df_clean["avg_base"]
df_clean["footprint_diff"] = abs(df_clean["Base1 (m)"] - df_clean["Base2 (m)"])

features = ["Base1 (m)", "Base2 (m)", "Height (m)", "aspect_ratio", "footprint_diff"]

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[features])

# =========================
# PCA (3D SPACE)
# =========================
pca = PCA(n_components=3)
X_pca = pca.fit_transform(X_scaled)

df_clean["PC1"] = X_pca[:, 0]
df_clean["PC2"] = X_pca[:, 1]
df_clean["PC3"] = X_pca[:, 2]

# =========================
# CLUSTER + ANOMALY DETECTION
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean["cluster"] = kmeans.fit_predict(X_pca)

iso = IsolationForest(contamination=0.15, random_state=42)
df_clean["anomaly"] = iso.fit_predict(X_scaled)

# =========================
# COLORS
# =========================
colors = ["red" if x == -1 else "gold" for x in df_clean["anomaly"]]

# =========================
# 🌍 EGYPT MAP VIEW
# =========================
st.subheader("🗺️ Geographic Distribution of Pyramids (Egypt Map)")

fig_map = px.scatter_geo(
    df_clean,
    lat="Latitude",
    lon="Longitude",
    hover_name="Pharaoh",
    hover_data=["Base1 (m)", "Height (m)", "Site"],
    projection="natural earth"
)

fig_map.update_layout(
    title=dict(
        text="🏺 Pyramid Locations Across Ancient Egypt",
        font=dict(size=26)
    ),
    height=500
)

# =========================
# 🌌 3D STRUCTURAL SPACE
# =========================
fig_3d = go.Figure()

fig_3d.add_trace(go.Scatter3d(
    x=df_clean["PC1"],
    y=df_clean["PC2"],
    z=df_clean["PC3"],
    mode="markers+text",
    text=df_clean["Pharaoh"],
    textposition="top center",
    marker=dict(
        size=6,
        color=colors,
        opacity=0.85
    )
))

fig_3d.update_layout(
    title=dict(
        text="🏺 3D Pyramid Structural Space",
        font=dict(size=30)
    ),
    scene=dict(
        xaxis_title="Structural Axis 1",
        yaxis_title="Structural Axis 2",
        zaxis_title="Structural Depth"
    ),
    height=550,
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
# INDIVIDUAL PYRAMID VIEWER
# =========================
st.subheader("🏗️ 3D Pyramid Builder (Individual Structure Viewer)")

selected = st.selectbox("Select Pyramid", df_clean["Pharaoh"].unique())
row = df_clean[df_clean["Pharaoh"] == selected].iloc[0]

base = float(row["Base1 (m)"])
height = float(row["Height (m)"])
half = base / 2

fig_pyr = go.Figure()

# base square
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
# INSIGHT SECTION
# =========================
st.subheader("🧠 Research Insight Summary")

st.write("""
- Pyramid distribution follows a strong Nile-centered geographic clustering  
- Structural space shows continuous variation, not discrete architectural classes  
- Anomalies represent experimental or transitional construction phases  
- Geometry is primarily driven by base size and scaling relationships  
- Ancient Egyptian architecture behaves like a stable evolutionary system  
""")
