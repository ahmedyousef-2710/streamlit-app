import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.set_page_config(page_title="Car Features Clustering App", layout="centered")

st.title("Car Features Clustering App")

st.markdown("""
This app applies **K-Means clustering** on car features and visualizes
the clusters using **PCA**.
""")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a CSV file with car features",
    type=["csv"]
)

if uploaded_file is not None:

    try:
        # 🔴 VERY IMPORTANT: reset file pointer
        uploaded_file.seek(0)

        # 🔹 Auto-detect separator and handle bad CSVs
        try:
            data = pd.read_csv(uploaded_file, sep=None, engine="python")
        except pd.errors.EmptyDataError:
            st.error("❌ The uploaded CSV file is empty or invalid.")
            st.stop()

        # 🔹 Validate CSV structure
        if data.empty or data.shape[1] == 0:
            st.error("❌ CSV file contains no usable columns.")
            st.stop()

        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        # Select numeric columns
        numeric_columns = data.select_dtypes(include="number").columns.tolist()

        if not numeric_columns:
            st.warning("⚠ No numeric columns found in the dataset.")
            st.stop()

        selected_features = st.multiselect(
            "Select numeric features for clustering",
            numeric_columns
        )

        if len(selected_features) < 2:
            st.warning("⚠ Please select at least two numeric features.")
            st.stop()

        # Prepare data
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

        # Number of clusters
        k = st.slider("Number of clusters (k)", 2, 10, 3)

        # KMeans clustering
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        clusters = kmeans.fit_predict(X_pca)

        # Attach clusters to original data
        data = data.loc[X.index]
        data["Cluster"] = clusters

        # Silhouette score
        score = silhouette_score(X_pca, clusters)
        st.success(f"Silhouette Score: {round(score, 3)}")

        # Plot clusters
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

        st.subheader("Clustered Data")
        st.dataframe(data)

    except Exception as e:
        st.error("❌ Failed to process the CSV file.")
        st.exception(e)
