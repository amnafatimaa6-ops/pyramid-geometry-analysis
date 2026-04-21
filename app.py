import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(page_title="Pyramid Cinematic Atlas", layout="wide")

st.title("🎬 Ancient Egypt: Cinematic Pyramid Intelligence System")

st.markdown("""
> “Along a single river, one of history’s greatest engineering civilizations emerged.  
> This is not random geography — this is structured intelligence shaped by water, survival, and time.”
""")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("pyramids.csv")
df.columns = df.columns.str.strip()

df = df.dropna(subset=["Latitude", "Longitude", "Base1 (m)", "Base2 (m)", "Height (m)"])

# =========================
# FEATURES
# =========================
df["avg_base"] = (df["Base1 (m)"] + df["Base2 (m)"]) / 2
df["aspect_ratio"] = df["Height (m)"] / df["avg_base"]

features = ["Base1 (m)", "Base2 (m)", "Height (m)", "aspect_ratio"]

X = StandardScaler().fit_transform(df[features])

# =========================
# PCA (3D)
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
# 🌊 NILE RIVER (CURVE)
# =========================
nile_lat = np.linspace(30.6, 24.8, 25)
nile_lon = np.linspace(31.25, 31.02, 25)

# =========================
# 🗺️ MAP VIEW (FIXED SAFE LAYOUT)
# =========================
fig_map = go.Figure()

fig_map.add_trace(go.Scattergeo(
    lat=df["Latitude"],
    lon=df["Longitude"],
    mode="markers",
    marker=dict(
        size=7,
        color=df["anomaly"],
        colorscale="Hot",
        opacity=0.85
    ),
    text=df["Pharaoh"]
))

fig_map.add_trace(go.Scattergeo(
    lat=nile_lat,
    lon=nile_lon,
    mode="lines",
    line=dict(color="cyan", width=3),
    name="Nile River"
))

fig_map.update_geos(
    projection_type="natural earth",
    showland=True,
    landcolor="rgb(20,20,20)",
    showocean=True,
    oceancolor="rgb(0,30,60)",
    showcountries=True,
    showcoastlines=True,
    center=dict(lat=26.8, lon=31.0)
)

fig_map.update_layout(
    paper_bgcolor="black",
    height=600,
    margin=dict(l=0, r=0, t=40, b=0),
    title="🏺 Egypt Civilization Map with Nile River"
)

# =========================
# 🌌 3D STRUCTURE SPACE
# =========================
fig_3d = go.Figure()

colors = ["red" if x == -1 else "gold" for x in df["anomaly"]]

fig_3d.add_trace(go.Scatter3d(
    x=df["PC1"],
    y=df["PC2"],
    z=df["PC3"],
    mode="markers",
    marker=dict(size=5, color=colors, opacity=0.8),
    text=df["Pharaoh"]
))

fig_3d.update_layout(
    title="🌌 3D Pyramid Structural Space",
    paper_bgcolor="black",
    scene=dict(
        xaxis_title="PC1",
        yaxis_title="PC2",
        zaxis_title="PC3"
    ),
    height=600
)

# =========================
# SIDE BY SIDE
# =========================
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    st.plotly_chart(fig_3d, use_container_width=True)

# =========================
# 🏗️ PYRAMID BUILDER
# =========================
st.subheader("🏗️ 3D Pyramid Builder")

selected = st.selectbox("Select Pyramid", df["Pharaoh"].unique())
row = df[df["Pharaoh"] == selected].iloc[0]

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
    height=600,
    scene=dict(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        zaxis=dict(visible=False)
    )
)

st.plotly_chart(fig_pyr, use_container_width=True)

# =========================
# 🧠 INSIGHTS
# =========================
st.subheader("🧠 Key Insight")

st.write("""
- Pyramids follow a strong Nile-based spatial structure  
- Geometry is consistent across dynasties  
- Anomalies represent architectural experimentation  
- Civilization is not random — it is geographically engineered  
""")
