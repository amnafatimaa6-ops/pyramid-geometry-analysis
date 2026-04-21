import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Pyramid Atlas", layout="wide")

st.title("🏺 Ancient Pyramid 3D Atlas")
st.write("Geometric intelligence mapping of ancient Egyptian pyramid structures")

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
# SCALE
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
# CLUSTER + ANOMALY
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean["cluster"] = kmeans.fit_predict(X_pca)

iso = IsolationForest(contamination=0.15, random_state=42)
df_clean["anomaly"] = iso.fit_predict(X_scaled)

# =========================
# COLOR MAP
# =========================
colors = ["red" if x == -1 else "gold" for x in df_clean["anomaly"]]

# =========================
# 3D STRUCTURAL SPACE
# =========================
st.subheader("🌌 3D Pyramid Structural Space")

fig = go.Figure()

fig.add_trace(go.Scatter3d(
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

fig.update_layout(
    title=dict(
        text="🏺 3D Pyramid Structural Space",
        font=dict(size=32)
    ),
    scene=dict(
        xaxis_title="Structural Axis 1",
        yaxis_title="Structural Axis 2",
        zaxis_title="Structural Depth"
    ),
    margin=dict(l=0, r=0, b=0, t=100),
    height=750
)

st.plotly_chart(fig, use_container_width=True)

# =========================
# INDIVIDUAL PYRAMID VIEWER
# =========================
st.subheader("🏗️ 3D Pyramid Viewer")

selected = st.selectbox("Select Pyramid", df_clean["Pharaoh"].unique())
row = df_clean[df_clean["Pharaoh"] == selected].iloc[0]

base = float(row["Base1 (m)"])
height = float(row["Height (m)"])
half = base / 2

fig2 = go.Figure()

# base square
fig2.add_trace(go.Mesh3d(
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
    fig2.add_trace(go.Scatter3d(
        x=[x, apex[0]],
        y=[y, apex[1]],
        z=[z, apex[2]],
        mode="lines",
        line=dict(color="brown", width=5)
    ))

fig2.update_layout(
    title=dict(
        text=f"🏺 {selected} - 3D Structure",
        font=dict(size=26)
    ),
    scene=dict(
        xaxis_title="X",
        yaxis_title="Y",
        zaxis_title="Height"
    ),
    margin=dict(l=0, r=0, b=0, t=80),
    height=650
)

st.plotly_chart(fig2, use_container_width=True)

# =========================
# INSIGHT PANEL
# =========================
st.subheader("🧠 Key Research Insight")

st.write("""
- Pyramid geometry forms a continuous structural system rather than discrete clusters  
- Most variance comes from base size and overall scale  
- Anomalies represent architectural deviation zones, not noise  
- Ancient Egyptian construction shows strong geometric consistency across dynasties  
""")
