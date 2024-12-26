import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Variabel Global
uploaded_data = None
processed_data = None
original_data = None
clustering_labels = None

# Fungsi untuk menampilkan data dalam tabel
def display_data(data):
    st.dataframe(data)

# Fungsi utama aplikasi
def main():
    global uploaded_data, processed_data, original_data, clustering_labels

    st.title("Clustering Analysis Web App")

    # Tab Panel
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "1. Upload Data",
        "2. Preprocessing & Elbow Method",
        "3. Clustering",
        "4. Evaluation",
        "5. Relabel Clusters"
    ])

    # Tab 1: Upload Data
    with tab1:
        st.header("Upload CSV File atau Gunakan Data Default")

        uploaded_file = st.file_uploader("Upload CSV File", type="csv")
        use_default = st.checkbox("Gunakan Data Default")

        if uploaded_file is not None:
            try:
                uploaded_data = pd.read_csv(uploaded_file)
                original_data = uploaded_data.copy()
                st.success("Data berhasil dimuat dari file upload.")
                display_data(uploaded_data)
            except Exception as e:
                st.error(f"Gagal memuat data: {e}")

        elif use_default:
            try:
                uploaded_data = pd.read_csv("case1.csv")  # Ganti dengan path file default Anda
                original_data = uploaded_data.copy()
                st.success("Data default berhasil dimuat.")
                display_data(uploaded_data)
            except Exception as e:
                st.error(f"Gagal memuat data default: {e}")

    # Tab 2: Preprocessing & Elbow Method
    with tab2:
        st.header("Preprocessing dan Elbow Method")

        if uploaded_data is not None:
            numeric_columns = uploaded_data.select_dtypes(include=["float64", "int64"]).columns

            if st.button("Standarisasi Data"):
                try:
                    scaler = StandardScaler()
                    processed_data = scaler.fit_transform(uploaded_data[numeric_columns])
                    st.success("Data berhasil distandarisasi.")
                    display_data(pd.DataFrame(processed_data, columns=numeric_columns))
                except Exception as e:
                    st.error(f"Gagal standarisasi data: {e}")

            if st.button("Lihat Grafik Elbow"):
                if processed_data is not None:
                    try:
                        sse = []
                        k_range = range(1, 11)
                        for k in k_range:
                            kmeans = KMeans(n_clusters=k, random_state=42)
                            kmeans.fit(processed_data)
                            sse.append(kmeans.inertia_)

                        fig, ax = plt.subplots()
                        ax.plot(k_range, sse, marker='o')
                        ax.set_title("Metode Elbow")
                        ax.set_xlabel("Jumlah Klaster (k)")
                        ax.set_ylabel("Sum of Squared Errors (SSE)")
                        st.pyplot(fig)
                    except Exception as e:
                        st.error(f"Gagal menghasilkan grafik Elbow: {e}")
                else:
                    st.error("Data belum diproses. Standarisasi data terlebih dahulu.")

    # Tab 3: Clustering
    with tab3:
        st.header("Clustering")

        if processed_data is not None:
            k = st.slider("Pilih jumlah klaster (k):", min_value=2, max_value=10, value=3)
            if st.button("Jalankan K-Means"):
                try:
                    kmeans = KMeans(n_clusters=k, random_state=42)
                    clustering_labels = kmeans.fit_predict(processed_data)
                    original_data["Cluster"] = clustering_labels
                    st.success(f"Clustering selesai dengan k={k}")
                    display_data(original_data)
                except Exception as e:
                    st.error(f"Gagal melakukan clustering: {e}")
        else:
            st.error("Data belum diproses. Lakukan preprocessing terlebih dahulu.")

    # Tab 4: Evaluation
    with tab4:
        st.header("Evaluation")

        if clustering_labels is not None:
            if st.button("Hitung Silhouette Score"):
                try:
                    silhouette_avg = silhouette_score(processed_data, clustering_labels)
                    st.write(f"Silhouette Score: {silhouette_avg:.2f}")
                except Exception as e:
                    st.error(f"Gagal menghitung Silhouette Score: {e}")

            if st.button("Visualisasi Klaster 2D"):
                try:
                    pca = PCA(n_components=2)
                    reduced_data = pca.fit_transform(processed_data)
                    fig, ax = plt.subplots()
                    scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clustering_labels, cmap="viridis", s=50)
                    ax.set_title("Visualisasi Klaster 2D")
                    ax.set_xlabel("Komponen Utama 1")
                    ax.set_ylabel("Komponen Utama 2")
                    fig.colorbar(scatter, label="Klaster")
                    st.pyplot(fig)
                except Exception as e:
                    st.error(f"Gagal visualisasi klaster 2D: {e}")

    # Tab 5: Relabel Clusters
    with tab5:
        st.header("Relabel Clusters")

        if clustering_labels is not None:
            st.write("Masukkan label baru untuk setiap klaster:")
            unique_labels = sorted(set(clustering_labels))
            new_labels = {}

            for cluster in unique_labels:
                new_label = st.text_input(f"Label untuk Cluster {cluster}:", f"Cluster {cluster}")
                new_labels[cluster] = new_label

            if st.button("Update Labels"):
                try:
                    original_data["Cluster"] = original_data["Cluster"].map(new_labels)
                    st.success("Label klaster berhasil diperbarui.")
                    display_data(original_data)
                except Exception as e:
                    st.error(f"Gagal memperbarui label: {e}")

            if st.button("Download Data Clustered (CSV)"):
                try:
                    csv = original_data.to_csv(index=False)
                    st.download_button(
                        label="Download Data",
                        data=csv,
                        file_name="clustered_data.csv",
                        mime="text/csv"
                    )
                except Exception as e:
                    st.error(f"Gagal mengunduh data: {e}")

if __name__ == "__main__":
    main()
