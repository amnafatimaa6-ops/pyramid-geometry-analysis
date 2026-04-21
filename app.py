import pandas as pd
import numpy as np
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("pyramids.csv")

# =========================
# CLEANING
# =========================
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

df.replace(["NaN", "nan", "-", "?"], np.nan, inplace=True)

numeric_cols = ['base1_(m)', 'base2_(m)', 'height_(m)', 'slope_(dec_degr)', 'volume_(cu.m)']
for col in numeric_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Fill slope
df['slope_(dec_degr)'] = df['slope_(dec_degr)'].fillna(
    np.degrees(np.arctan(df['height_(m)'] / ((df['base1_(m)'] + df['base2_(m)']) / 4)))
)

# Clean text
for col in ['pharaoh', 'ancient_name', 'modern_name', 'site', 'type']:
    df[col] = df[col].astype(str).str.lower().str.strip()

# =========================
# FEATURE ENGINEERING
# =========================
df['avg_base'] = (df['base1_(m)'] + df['base2_(m)']) / 2
df['aspect_ratio'] = df['height_(m)'] / df['avg_base']
df['footprint_diff'] = abs(df['base1_(m)'] - df['base2_(m)'])

# =========================
# ML PREP
# =========================
features = ['base1_(m)', 'base2_(m)', 'height_(m)', 'aspect_ratio', 'footprint_diff']
ml_df = df[features].dropna()

scaler = StandardScaler()
X_scaled = scaler.fit_transform(ml_df)

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_pca = pd.DataFrame(X_pca, columns=["PC1", "PC2"])

# Clustering
kmeans = KMeans(n_clusters=3, random_state=42)
df_pca["cluster"] = kmeans.fit_predict(X_pca)

# Anomaly detection
iso = IsolationForest(contamination=0.15, random_state=42)
df_pca["anomaly"] = iso.fit_predict(X_scaled)

# =========================
# 🌊 NILE CURVE
# =========================
nile_lat = np.linspace(30, 24, 200)
nile_lon = 31 + 0.8*np.sin(np.linspace(0, 3*np.pi, 200))

# =========================
# 🌍 MAP (FIXED - NO BLACK SCREEN)
# =========================
fig_map = go.Figure()

# Pyramids
fig_map.add_trace(go.Scattergeo(
    lat=df["latitude"],
    lon=df["longitude"],
    mode="markers",
    marker=dict(size=8, color="red"),
    text=df["pharaoh"],
    name="Pyramids"
))

# Nile line
fig_map.add_trace(go.Scattergeo(
    lat=nile_lat,
    lon=nile_lon,
    mode="lines",
    line=dict(width=4, color="blue"),
    name="Nile River"
))

# Shimmer effect (SAFE)
fig_map.add_trace(go.Scattergeo(
    lat=nile_lat,
    lon=nile_lon,
    mode="markers",
    marker=dict(size=6, color="cyan", opacity=0.4),
    showlegend=False
))

fig_map.update_layout(
    title="🌍 Egypt Pyramid Map",
    geo=dict(
        scope="africa",
        projection_type="natural earth",
        showland=True,
        landcolor="rgb(240,230,200)",
        showcountries=True,
        lataxis_range=[20, 35],
        lonaxis_range=[25, 35]
    )
)

fig_map.show()

# =========================
# 🏜️ 3D PYRAMID SPACE (FIXED)
# =========================
fig_geo3d = go.Figure()

fig_geo3d.add_trace(go.Scatter3d(
    x=df["longitude"],
    y=df["latitude"],
    z=df["height_(m)"].fillna(0),  # FIX
    mode="markers",
    marker=dict(size=5, color=df["height_(m)"].fillna(0), colorscale="Viridis"),
    text=df["pharaoh"]
))

# 3D Nile
fig_geo3d.add_trace(go.Scatter3d(
    x=nile_lon,
    y=nile_lat,
    z=[0]*len(nile_lat),
    mode="lines",
    line=dict(color="blue", width=6)
))

# 3D shimmer
fig_geo3d.add_trace(go.Scatter3d(
    x=nile_lon,
    y=nile_lat,
    z=[2]*len(nile_lat),
    mode="markers",
    marker=dict(size=2, color="cyan", opacity=0.4)
))

fig_geo3d.update_layout(
    title="🏜️ 3D Pyramid Structural Space",
    height=700
)

fig_geo3d.show()

# =========================
# 🏺 DYNASTY TIMELINE
# =========================
df["dynasty"] = pd.to_numeric(df["dynasty"], errors="coerce")

timeline = df.groupby("dynasty").size().reset_index(name="count")

fig_timeline = go.Figure()

fig_timeline.add_trace(go.Scatter(
    x=timeline["dynasty"],
    y=timeline["count"],
    mode="lines+markers"
))

fig_timeline.update_layout(
    title="🏺 Pyramid Evolution Across Dynasties",
    xaxis_title="Dynasty",
    yaxis_title="Count",
    height=500
)

fig_timeline.show()

# =========================
# 📈 HEIGHT EVOLUTION
# =========================
fig_height = go.Figure()

fig_height.add_trace(go.Scatter(
    x=df["dynasty"],
    y=df["height_(m)"],
    mode="markers",
    marker=dict(
        size=10,
        color=df["height_(m)"],
        colorscale="Viridis",
        showscale=True
    )
))

fig_height.update_layout(
    title="📈 Pyramid Height Evolution",
    xaxis_title="Dynasty",
    yaxis_title="Height (m)",
    height=500
)

fig_height.show()
