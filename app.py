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
# PAGE CONFIG
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
# CLEAN + FEATURE ENGINEERING
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
# 🌊 NILE RIVER PATH
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
# 🗺️ MAP VIEW
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
- Geometry is consistent with small anomalies  
- Variation reflects evolution, not randomness  
""")

# ==========================================================
# 🏺 FIXED ANIMATED EVOLUTION (BC TIMELINE)
# ==========================================================
st.subheader("🏺 Pyramid Evolution Across Ancient Egypt Timeline")

df_anim = df_clean.copy()
df_anim["Dynasty"] = pd.to_numeric(df_anim["Dynasty"], errors="coerce")
df_anim = df_anim.dropna(subset=["Dynasty"])
df_anim["Dynasty"] = df_anim["Dynasty"].astype(int)

def dynasty_to_period(d):
    if d <= 3:
        return "Early Dynastic (2700–2600 BC)"
    elif d <= 6:
        return "Old Kingdom (2600–2200 BC)"
    elif d <= 11:
        return "First Intermediate (2200–2050 BC)"
    elif d <= 12:
        return "Middle Kingdom (2050–1700 BC)"
    elif d <= 17:
        return "Second Intermediate (1700–1550 BC)"
    else:
        return "New Kingdom (1550–1070 BC)"

df_anim["Period"] = df_anim["Dynasty"].apply(dynasty_to_period)

order = [
    "Early Dynastic (2700–2600 BC)",
    "Old Kingdom (2600–2200 BC)",
    "First Intermediate (2200–2050 BC)",
    "Middle Kingdom (2050–1700 BC)",
    "Second Intermediate (1700–1550 BC)",
    "New Kingdom (1550–1070 BC)"
]

df_anim["Period"] = pd.Categorical(df_anim["Period"], categories=order, ordered=True)

df_anim["size"] = df_anim["Height (m)"] / df_anim["Height (m)"].max()

fig_evo = px.scatter_3d(
    df_anim,
    x="Longitude",
    y="Latitude",
    z="Height (m)",
    color="Period",
    animation_frame="Period",
    hover_name="Pharaoh",
    size="size",
    size_max=10,
    title="🏺 Egypt Pyramid Evolution (BC Timeline)"
)

fig_evo.update_layout(height=700)

st.plotly_chart(fig_evo, use_container_width=True)
