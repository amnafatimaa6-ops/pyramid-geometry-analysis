import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.ensemble import IsolationForest

st.set_page_config(page_title="Pyramid Geometry Analysis", layout="wide")

# =========================
# LOAD DATA
# =========================
df = pd.read_csv("pyramids.csv")

st.title("🏺 Ancient Pyramid Geometry Analysis")

st.subheader("Dataset Preview")
st.write(df.head())

# =========================
# CLEAN + FEATURES (same as notebook)
# =========================
df = df.copy()

df.columns = df.columns.str.strip()

features = ['Base1 (m)', 'Base2 (m)', 'Height (m)']

df_clean = df.dropna(subset=features)

df_clean["avg_base"] = (df_clean['Base1 (m)'] + df_clean['Base2 (m)']) / 2
df_clean["aspect_ratio"] = df_clean['Height (m)'] / df_clean["avg_base"]
df_clean["footprint_diff"] = abs(df_clean['Base1 (m)'] - df_clean['Base2 (m)'])

ml_features = ['Base1 (m)', 'Base2 (m)', 'Height (m)', 'aspect_ratio', 'footprint_diff']

X = df_clean[ml_features]

# =========================
# SCALING
# =========================
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# =========================
# PCA
# =========================
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

df_clean["PC1"] = X_pca[:, 0]
df_clean["PC2"] = X_pca[:, 1]

# =========================
# KMEANS CLUSTERING
# =========================
kmeans = KMeans(n_clusters=3, random_state=42)
df_clean["cluster"] = kmeans.fit_predict(X_pca)

# =========================
# ANOMALY DETECTION
# =========================
iso = IsolationForest(contamination=0.15, random_state=42)
df_clean["anomaly"] = iso.fit_predict(X_scaled)

# =========================
# VISUAL 1: CLUSTERS
# =========================
st.subheader("🧱 Pyramid Clusters (PCA Space)")

fig1, ax1 = plt.subplots()
scatter = ax1.scatter(df_clean["PC1"], df_clean["PC2"], c=df_clean["cluster"])
ax1.set_xlabel("PC1")
ax1.set_ylabel("PC2")
ax1.set_title("Structural Clusters")
st.pyplot(fig1)

# =========================
# VISUAL 2: ANOMALIES
# =========================
st.subheader("🕳️ Structural Anomalies")

fig2, ax2 = plt.subplots()
ax2.scatter(df_clean["PC1"], df_clean["PC2"], c=df_clean["anomaly"])
ax2.set_xlabel("PC1")
ax2.set_ylabel("PC2")
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
ax3.set_xlabel("Site")
ax3.set_title("Site Distribution")
plt.xticks(rotation=45)
st.pyplot(fig3)

# =========================
# CORRELATION HEATMAP
# =========================
st.subheader("📊 Correlation Heatmap")

fig4, ax4 = plt.subplots(figsize=(8, 6))
sns.heatmap(df[ml_features].corr(), annot=True, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)

# =========================
# FINAL INSIGHT
# =========================
st.subheader("🧠 Key Insight")

st.write("""
- Pyramid geometry is highly consistent across sites  
- No strong clustering → continuous structural variation  
- Few anomalies represent experimental or transitional designs  
- Structure is driven mainly by base geometry and scale  
""")
