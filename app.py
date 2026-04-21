import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# =========================
# PAGE SETUP
# =========================
st.set_page_config(page_title="Pyramid Atlas", layout="wide")

st.title("🏺 Ancient Egypt Pyramid Intelligence Atlas")
st.write("Geometric + Geographic + Historical AI Exploration System")

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
# SCALE DATA
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
# 🌊 NILE RIVER
# =========================
nile_lat = [30.6, 30.2, 29.9, 29.6, 29.3, 29.0, 28.7, 28.3, 27.9, 27.5]
nile_lon = [31.2, 31.25, 31.3, 31.28, 31.26, 31.24, 31.22, 31.20, 31.18, 31.16]

# =========================
# ZONES
# =========================
df_clean["zone"] = "other"

df_clean.loc[
    df_clean["Site"].str.contains("giza|saqqara|dahshur|abusir|lisht", case=False, na=False),
    "zone"
] = "major_pyramid_zone"

df_clean.loc[
    df_clean["Site"].str.contains("abydos|edfu|elephantine|hierakonpolis|dara", case=False, na=False),
    "zone"
] = "ancient_upper_egypt"

color_map = {
    "major_pyramid_zone": "gold",
    "ancient_upper_egypt": "orange",
    "other": "lightblue"
}

# =========================
# 🗺️ MAP
# =========================
st.subheader("🗺️ Egypt Map + Nile River")

fig_map = px.scatter_geo(
    df_clean,
    lat="Latitude",
    lon="Longitude",
    color="zone",
    hover_name="Pharaoh",
    hover_data=["Site", "Height (m)"],
    projection="natural earth"
)

fig_map.add_trace(go.Scattergeo(
    lat=nile_lat,
    lon=nile_lon,
    mode="lines",
    line=dict(width=3, color="blue"),
    name="Nile River"
))

fig_map.update_layout(height=600)

# =========================
# 🌌 3D STRUCTURAL SPACE
# =========================
st.subheader("🌌 3D Pyramid Structural Space")

fig_3d = go.Figure()

colors = ["red" if x == -1 else "gold" for x in df_clean["anomaly"]]

fig_3d.add_trace(go.Scatter3d(
    x=df_clean["PC1"],
    y=df_clean["PC2"],
    z=df_clean["PC3"],
    mode="markers",
    marker=dict(size=5, color=colors),
    text=df_clean["Pharaoh"]
))

fig_3d.update_layout(height=600)

col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig_map, use_container_width=True)

with col2:
    st.plotly_chart(fig_3d, use_container_width=True)

# =========================
# 🌊 3D GEOGRAPHY VIEW
# =========================
st.subheader("🌊 Nile + Pyramid Geography (3D)")

fig_geo = go.Figure()

fig_geo.add_trace(go.Scatter3d(
    x=nile_lon,
    y=nile_lat,
    z=[0]*len(nile_lat),
    mode="lines",
    line=dict(color="blue", width=5),
    name="Nile"
))

fig_geo.add_trace(go.Scatter3d(
    x=df_clean["Longitude"],
    y=df_clean["Latitude"],
    z=df_clean["Height (m)"],
    mode="markers",
    marker=dict(size=5, color="gold"),
    text=df_clean["Pharaoh"]
))

st.plotly_chart(fig_geo, use_container_width=True)

# =========================
# 🏗️ PYRAMID BUILDER
# =========================
st.subheader("🏗️ Pyramid Builder")

selected = st.selectbox("Select Pyramid", df_clean["Pharaoh"].unique())
row = df_clean[df_clean["Pharaoh"] == selected].iloc[0]

base = row["Base1 (m)"]
height = row["Height (m)"]
h = base / 2

fig_pyr = go.Figure()

fig_pyr.add_trace(go.Mesh3d(
    x=[-h, h, h, -h],
    y=[-h, -h, h, h],
    z=[0, 0, 0, 0],
    color="tan",
    opacity=0.5
))

apex = (0, 0, height)

for x, y in [(-h,-h), (h,-h), (h,h), (-h,h)]:
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
st.subheader("🧠 Insights")

st.write("""
- Pyramid construction follows Nile corridor  
- Giza–Saqqara–Dahshur dominate architecture  
- Geometry is consistent with few anomalies  
- Variation reflects historical evolution  
""")

# ==========================================================
# ⏳ BC → AD TIMELINE (ADDED BLOCK)
# ==========================================================
st.subheader("⏳ Egypt Civilization Timeline (BC → AD)")

timeline_data = pd.DataFrame({
    "Era": [
        "Early Dynastic",
        "Old Kingdom",
        "First Intermediate",
        "Middle Kingdom",
        "Second Intermediate",
        "New Kingdom",
        "Late Period",
        "Ptolemaic Era",
        "Roman Egypt",
        "Modern Egypt"
    ],
    "Start": [-3000, -2600, -2200, -2050, -1700, -1550, -700, -332, -30, 1800],
    "End":   [-2600, -2200, -2050, -1700, -1550, -1070, -332, -30, 395, 2026]
})

fig_time = go.Figure()

for i, row in timeline_data.iterrows():
    fig_time.add_trace(go.Scatter(
        x=[row["Start"], row["End"]],
        y=[row["Era"], row["Era"]],
        mode="lines+markers",
        line=dict(width=8),
        marker=dict(size=10),
        name=row["Era"]
    ))

fig_time.update_layout(
    title="🏺 Egypt Civilization Evolution Timeline",
    xaxis_title="Time (BC → AD)",
    yaxis_title="Era",
    height=600
)

st.plotly_chart(fig_time, use_container_width=True)
