import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

# =========================
# CONFIG
# =========================
st.set_page_config(page_title="Pyramid Explorer", layout="wide")

st.title("🏺 Ancient Pyramid 3D Geometry Explorer")
st.write("Interactive ML + 3D reconstruction of pyramid structures")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("pyramids.csv")

# clean column names (important fix)
df.columns = df.columns.str.strip()

st.subheader("📊 Dataset Preview")
st.dataframe(df.head())

# =========================
# FEATURE ENGINEERING
# =========================
features = ['Base1 (m)', 'Base2 (m)', 'Height (m)']
df_clean = df.dropna(subset=features).copy()

df_clean["avg_base"] = (df_clean["Base1 (m)"] + df_clean["Base2 (m)"]) / 2
df_clean["aspect_ratio"] = df_clean["Height (m)"] / df_clean["avg_base"]
df_clean["footprint_diff"] = abs(df_clean["Base1 (m)"] - df_clean["Base2 (m)"])

ml_features = ['Base1 (m)', 'Base2 (m)', 'Height (m)', 'aspect_ratio', 'footprint_diff']

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_clean[ml_features])

# =========================
# PCA
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_clean["PC1"] = X_pca[:, 0]
df_clean["PC2"] = X_pca[:, 1]

# =========================
# CLUSTERING
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean["cluster"] = kmeans.fit_predict(X_pca)

# =========================
# ANOMALY DETECTION
# =========================
iso = IsolationForest(contamination=0.15, random_state=42)
df_clean["anomaly"] = iso.fit_predict(X_scaled)

# =========================
# 3D PYRAMID FUNCTION
# =========================
def draw_pyramid(base, height):
    half = base / 2

    fig = go.Figure()

    # base square
    fig.add_trace(go.Mesh3d(
        x=[-half, half, half, -half],
        y=[-half, -half, half, half],
        z=[0, 0, 0, 0],
        color="tan",
        opacity=0.5
    ))

    # apex
    apex = (0, 0, height)

    base_points = [
        (-half, -half, 0),
        (half, -half, 0),
        (half, half, 0),
        (-half, half, 0)
    ]

    # pyramid sides
    for x, y, z in base_points:
        fig.add_trace(go.Scatter3d(
            x=[x, apex[0]],
            y=[y, apex[1]],
            z=[z, apex[2]],
            mode="lines",
            line=dict(color="brown", width=4)
        ))

    fig.update_layout(
        title="3D Pyramid Structure",
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Height"
        ),
        margin=dict(l=0, r=0, b=0, t=40)
    )

    return fig

# =========================
# INTERACTIVE 3D VIEW
# =========================
st.subheader("🏗️ 3D Pyramid Explorer")

selected = st.selectbox("Select Pyramid", df_clean["Pharaoh"].unique())

row = df_clean[df_clean["Pharaoh"] == selected].iloc[0]

st.write("### 📐 Pyramid Features")
st.write(row[["Base1 (m)", "Base2 (m)", "Height (m)"]])

fig3d = draw_pyramid(
    base=float(row["Base1 (m)"]),
    height=float(row["Height (m)"])
)

st.plotly_chart(fig3d, use_container_width=True)

# =========================
# PCA CLUSTERS
# =========================
st.subheader("🧱 Structural Clusters (PCA Space)")

fig1, ax1 = plt.subplots()
ax1.scatter(df_clean["PC1"], df_clean["PC2"], c=df_clean["cluster"])
ax1.set_title("Pyramid Clusters")
st.pyplot(fig1)

# =========================
# ANOMALIES
# =========================
st.subheader("🕳️ Structural Anomalies")

fig2, ax2 = plt.subplots()
ax2.scatter(df_clean["PC1"], df_clean["PC2"], c=df_clean["anomaly"])
ax2.set_title("Anomaly Detection")
st.pyplot(fig2)

# =========================
# SITE DISTRIBUTION
# =========================
st.subheader("📍 Pyramid Distribution by Site")

site_counts = df["Site"].value_counts()

fig3, ax3 = plt.subplots(figsize=(10, 5))
site_counts.plot(kind="bar", ax=ax3)
ax3.set_ylabel("Count")
ax3.set_title("Site Distribution")
plt.xticks(rotation=45)
st.pyplot(fig3)

# =========================
# CORRELATION HEATMAP
# =========================
st.subheader("📊 Correlation Heatmap")

fig4, ax4 = plt.subplots(figsize=(7, 5))
sns.heatmap(df[ml_features].corr(), annot=True, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)

# =========================
# INSIGHT PANEL
# =========================
st.subheader("🧠 Key Insight")

st.write("""
- Pyramid geometry is highly consistent across Egypt  
- No strong clustering → continuous architectural evolution  
- Few anomalies represent experimental or transitional designs  
- Structure is mainly driven by base geometry and scale  
""")
