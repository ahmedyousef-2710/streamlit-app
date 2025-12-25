import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Car Features Clustering App",
    layout="centered"
)

st.title("Car Features Clustering App")

st.markdown("""
This app applies **K-Means clustering** on car features and visualizes  
the clusters using **PCA**.
""")

# ---------------- File Upload ----------------
uploaded_file = st.file_uploader(
    "Upload a CSV file with car features",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # Reset pointer
        uploaded_file.seek(0)

        # Safe CSV loading (encoding + separator)
        try:
            data = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python",
                encoding="utf-8"
            )
        except UnicodeDecodeError:
            uploaded_file.seek(0)
            data = pd.read_csv(
                uploaded_file,
                sep=None,
                engine="python",
                encoding="latin1"
            )

        # Basic validation
        if data.empty or data.shape[1] == 0:
            st.error("❌ CSV file contains no usable data.")
            st.stop()

        # ---------------- Dataset Preview ----------------
        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        # ---------------- Numeric Columns ----------------
        numeric_columns = data.select_dtypes(include="number").columns.tolist()

        with st.expander("Detected numeric columns"):
            st.write(numeric_columns)

        if len(numeric_columns) < 2:
            st.error("❌ Dataset must contain at least two numeric columns.")
            st.stop()

        # ---------------- Feature Selection ----------------
        st.subheader("Feature Selection")

        selected_features = st.multiselect(
            "Select numeric features for clustering (at least 2)",
            numeric_columns,
            default=numeric_columns[:2]  # 👈 auto-select
        )

        if len(selected_features) < 2:
            st.warning("⚠ Please select at least two numeric features.")
            st.stop()

        st.success(f"Using features: {selected_features}")

        # ---------------- Prepare Data ----------------
        X = data[selected_features].dropna()

        if X.empty:
            st.error("❌ Selected features contain only missing values.")
            st.stop()

        # Scaling
        scaler = RobustScaler()
        X_scaled = scaler.fit_transform(X)

        # PCA for visualization
        pca = PCA(n_components=2)
        X_pca = pca.fit_transform(X_scaled)

        # ---------------- Clustering ----------------
        k = st.slider("Number of clusters (k)", 2, 10, 3)

        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_pca)

        # Attach clusters
        data = data.loc[X.index].copy()
        data["Cluster"] = clusters

        # ---------------- Evaluation ----------------
        score = silhouette_score(X_scaled, clusters)
        st.success(f"Silhouette Score: {round(score, 3)}")

        # ---------------- Visualization ----------------
        st.subheader("PCA Cluster Visualization")

        fig, ax = plt.subplots()
        ax.scatter(
            X_pca[:, 0],
            X_pca[:, 1],
            c=clusters,
            cmap="viridis"
        )
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.set_title("Car Clusters (PCA Projection)")
        st.pyplot(fig)

        # ---------------- Clustered Data ----------------
        st.subheader("Clustered Data")
        st.dataframe(data)

    except Exception as e:
        st.error("❌ Failed to process the CSV file.")
        st.exception(e)
