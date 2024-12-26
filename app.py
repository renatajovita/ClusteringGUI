import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Fungsi untuk menampilkan data dalam tabel
def display_data(data):
    st.write("Data Preview:")
    st.dataframe(data)

# Fungsi utama aplikasi
def main():
    st.title("Clustering Analysis Web App")
    st.write("""
    ### GUI CLUSTERING (KELOMPOK 17)
    Disusun oleh:
    - Alya Faadhila Rosyid (24050122130049)  
    - Mesakh Besta Anugrah (24050122130058)  
    - Renata Jovita Aurelia Dinda (24050122130094)  
    **Departemen Statistika, Fakultas Sains dan Matematika, Universitas Diponegoro, Semarang, 2024**
    """)

    # Pilihan data
    st.header("1. Upload Data atau Gunakan Data Default")
    uploaded_file = st.file_uploader("Upload CSV File", type="csv")
    use_default = st.checkbox("Gunakan Data Default")
    data = None

    # Load data
    if uploaded_file is not None:
        data = pd.read_csv(uploaded_file)
        st.success("Data berhasil dimuat dari file upload.")
    elif use_default:
        default_path = "case1.csv" 
        try:
            data = pd.read_csv(default_path)
            st.success("Data default berhasil dimuat.")
        except FileNotFoundError:
            st.error("File default tidak ditemukan. Pastikan file 'case1.csv' ada di repositori GitHub Anda.")

    if data is not None:
        display_data(data)

        # Preprocessing
        st.header("2. Preprocessing dan Elbow Method")
        scaler = StandardScaler()
        numeric_columns = data.select_dtypes(include=["float64", "int64"]).columns
        processed_data = scaler.fit_transform(data[numeric_columns])
        st.write("Data distandarisasi:")
        st.dataframe(pd.DataFrame(processed_data, columns=numeric_columns))

        # Elbow Method
        st.subheader("Metode Elbow")
        sse = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(processed_data)
            sse.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(k_range, sse, marker='o')
        ax.set_title("Elbow Method")
        ax.set_xlabel("Jumlah Klaster (k)")
        ax.set_ylabel("Sum of Squared Errors (SSE)")
        st.pyplot(fig)

        # Clustering
        st.header("3. Clustering")
        k = st.slider("Pilih jumlah klaster (k):", min_value=2, max_value=10, value=3)
        if st.button("Jalankan Clustering"):
            kmeans = KMeans(n_clusters=k, random_state=42)
            labels = kmeans.fit_predict(processed_data)
            data["Cluster"] = labels
            st.success(f"Clustering selesai dengan k={k}")
            st.write("Data dengan klaster:")
            st.dataframe(data)

        # Evaluation
        st.header("4. Evaluation")
        if "Cluster" in data.columns:
            silhouette_avg = silhouette_score(processed_data, data["Cluster"])
            st.write(f"Silhouette Score: {silhouette_avg:.2f}")

            # Visualisasi 2D
            st.subheader("Visualisasi Klaster 2D")
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(processed_data)
            fig, ax = plt.subplots()
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=data["Cluster"], cmap="viridis", s=50)
            ax.set_title("Visualisasi Klaster 2D")
            ax.set_xlabel("Komponen Utama 1")
            ax.set_ylabel("Komponen Utama 2")
            fig.colorbar(scatter, label="Klaster")
            st.pyplot(fig)

            # Visualisasi 3D
            st.subheader("Visualisasi Klaster 3D")
            pca_3d = PCA(n_components=3)
            reduced_data_3d = pca_3d.fit_transform(processed_data)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data_3d[:, 0], reduced_data_3d[:, 1], reduced_data_3d[:, 2],
                                  c=data["Cluster"], cmap="viridis", s=50)
            ax.set_title("Visualisasi Klaster 3D")
            ax.set_xlabel("Komponen Utama 1")
            ax.set_ylabel("Komponen Utama 2")
            ax.set_zlabel("Komponen Utama 3")
            fig.colorbar(scatter, label="Klaster")
            st.pyplot(fig)

        # Relabel Clusters
        st.header("5. Relabel Clusters")
        if "Cluster" in data.columns:
            st.write("Masukkan label baru untuk setiap klaster:")
            unique_clusters = sorted(data["Cluster"].unique())
            new_labels = {}
            for cluster in unique_clusters:
                new_label = st.text_input(f"Label untuk Cluster {cluster}:", f"Cluster {cluster}")
                new_labels[cluster] = new_label

            if st.button("Update Labels"):
                data["Cluster"] = data["Cluster"].map(new_labels)
                st.success("Label klaster berhasil diperbarui!")
                st.write("Data dengan label baru:")
                st.dataframe(data)

        # Export Data
        st.header("6. Export Data")
        if st.button("Download Data Clustered (CSV)"):
            csv = data.to_csv(index=False)
            st.download_button(
                label="Download Data Clustered",
                data=csv,
                file_name="clustered_data.csv",
                mime="text/csv"
            )

if __name__ == "__main__":
    main()


