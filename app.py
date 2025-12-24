import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

st.title("Car Features Clustering App")

st.markdown(""" 
This app applies clustering on car features
""")

# File uploader
uploaded_file = st.file_uploader(
    "Upload a CSV file with car features",
    type=["csv"]
)

if uploaded_file is not None:
    try:
        # Read CSV safely
        data = pd.read_csv(uploaded_file)


        st.subheader("Dataset Preview")
        st.dataframe(data.head())

        # Select numeric columns
        numeric_columns = data.select_dtypes(include="number").columns.tolist()

        if not numeric_columns:
            st.warning("No numeric columns found in the dataset.")
        else:
            selected_features = st.multiselect(
                "Select numeric features for clustering",
                numeric_columns
            )

            if len(selected_features) >= 2:
                X = data[selected_features]

                # Scaling
                scaler = RobustScaler()
                X_scaled = scaler.fit_transform(X)

                # PCA for 2D visualization
                pca = PCA(n_components=2)
                X_pca = pca.fit_transform(X_scaled)

                # Number of clusters
                k = st.slider("Number of clusters (k)", 2, 10, 3)

                # KMeans clustering
                kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
                clusters = kmeans.fit_predict(X_pca)

                data["Cluster"] = clusters

                # Silhouette score
                score = silhouette_score(X_pca, clusters)
                st.write("Silhouette Score:", round(score, 3))

                # Plot
                fig, ax = plt.subplots()
                scatter = ax.scatter(
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

            else:
                st.warning("Please select at least two numeric features.")

    except Exception as e:
        st.error(f"Error loading CSV file: {e}")
