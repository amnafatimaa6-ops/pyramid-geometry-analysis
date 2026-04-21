import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Pyramid Atlas", layout="wide")

st.title("🏺 Ancient Egypt Pyramid Intelligence Atlas")
st.write("Geometric + Geographic + Archaeological AI Exploration System")

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
# PCA (3D)
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
# 🌊 NILE RIVER
# =========================
nile_lat = [30.6, 30.2, 29.9, 29.6, 29.3, 29.0, 28.7, 28.3, 27.9, 27.5, 27.1, 26.7, 26.3, 25.9, 25.5, 25.1, 24.8]
nile_lon = [31.2, 31.25, 31.3, 31.28, 31.26, 31.24, 31.22, 31.20, 31.18, 31.16, 31.14, 31.12, 31.10, 31.08, 31.06, 31.04, 31.02]

# =========================
# MAP ZONES
# =========================
df_clean["zone"] = "other"

df_clean.loc[
    df_clean["Site"].str.contains("giza|saqqara|dahshur|abusir|lisht|hawara|el-lahun", case=False, na=False),
    "zone"
] = "major_pyramid_zone"

df_clean.loc[
    df_clean["Site"].str.contains("abydos|edfu|elephantine|hierakonpolis|dara", case=False, na=False),
    "zone"
] = "ancient_upper_egypt_zone"

color_map = {
    "major_pyramid_zone": "gold",
    "ancient_upper_egypt_zone": "orange",
    "other": "lightblue"
}

# =========================
# 🗺️ MAP
# =========================
st.subheader("🗺️ Egypt Civilization Map with Nile River")

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

fig_map.add_trace(go.Scattergeo(
    lat=nile_lat,
    lon=nile_lon,
    mode="lines",
    line=dict(width=3, color="blue"),
    name="Nile River"
))

fig_map.update_layout(height=650)

# =========================
# 🌌 3D STRUCTURE SPACE
# =========================
st.subheader("🌌 3D Pyramid Structural Space")

colors = ["red" if x == -1 else "gold" for x in df_clean["anomaly"]]

fig_3d = go.Figure()

fig_3d.add_trace(go.Scatter3d(
    x=df_clean["PC1"],
    y=df_clean["PC2"],
    z=df_clean["PC3"],
    mode="markers+text",
    text=df_clean["Pharaoh"],
    marker=dict(size=5, color=colors)
))

fig_3d.update_layout(height=600)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    st.plotly_chart(fig_3d, use_container_width=True)

# =========================
# 🌊 3D NILE VIEW
# =========================
st.subheader("🌊 3D Nile + Pyramid Geography")

fig_geo3d = go.Figure()

fig_geo3d.add_trace(go.Scatter3d(
    x=nile_lon,
    y=nile_lat,
    z=[0]*len(nile_lat),
    mode="lines",
    line=dict(color="blue", width=6),
    name="Nile"
))

fig_geo3d.add_trace(go.Scatter3d(
    x=df_clean["Longitude"],
    y=df_clean["Latitude"],
    z=df_clean["Height (m)"],
    mode="markers+text",
    text=df_clean["Pharaoh"],
    marker=dict(size=5, color="gold")
))

fig_geo3d.update_layout(height=600)

st.plotly_chart(fig_geo3d, use_container_width=True)

# =========================
# 🏗️ PYRAMID BUILDER
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
    opacity=0.4
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

st.plotly_chart(fig_pyr, use_container_width=True)

# =========================
# 🧠 INSIGHTS
# =========================
st.subheader("🧠 Key Insights")

st.write("""
- Pyramid construction follows the Nile corridor  
- Giza–Saqqara–Dahshur are major engineering hubs  
- Geometry is mostly consistent with few anomalies  
- Variation reflects historical evolution, not randomness  
""")

# =========================
# 🕰️ FIXED SEABORN TIMELINE (BC → AD)
# =========================
st.subheader("🕰️ Egypt Civilization Timeline (BC → AD)")

timeline_data = pd.DataFrame({
    "Era": [
        "Prehistoric", "Early Dynastic", "Old Kingdom",
        "First Intermediate", "Middle Kingdom", "Second Intermediate",
        "New Kingdom", "Late Period", "Ptolemaic", "Roman",
        "Islamic", "Modern Egypt"
    ],
    "Start": [-6000, -3100, -2686, -2181, -2055, -1650, -1550, -664, -332, -30, 640, 1800],
    "End":   [-3100, -2686, -2181, -2055, -1650, -1550, -664, -332, -30, 640, 1800, 2026]
})

fig, ax = plt.subplots(figsize=(10, 6))

for i, row in timeline_data.iterrows():
    ax.barh(
        row["Era"],
        row["End"] - row["Start"],
        left=row["Start"],
        color=sns.color_palette("viridis", len(timeline_data))[i]
    )

ax.axvline(0, color="black", linestyle="--", linewidth=1)
ax.set_xlabel("Time (BC → AD)")
ax.set_title("Egypt Civilization Evolution Timeline")

plt.tight_layout()
st.pyplot(fig)
