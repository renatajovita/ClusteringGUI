import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
import base64

# Fungsi untuk mengunduh DataFrame sebagai CSV
def download_csv(dataframe, filename="clustered_data.csv"):
    csv = dataframe.to_csv(index=False)
    b64 = base64.b64encode(csv.encode()).decode()  # Encoding ke Base64
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Klik di sini untuk mengunduh file CSV</a>'
    return href

# Konfigurasi Streamlit
st.set_page_config(page_title="Clustering Analysis App", layout="wide")

# Header aplikasi
st.title("Clustering Analysis App")
st.write("Analisis clustering dengan fitur unggulan seperti preprocessing, elbow method, clustering, dan evaluasi.")

# Panel untuk mengunggah data
st.sidebar.header("1. Upload Data")
uploaded_file = st.sidebar.file_uploader("Unggah file CSV Anda", type=["csv"])

# Memuat data default jika pengguna tidak mengunggah file
if uploaded_file is not None:
    data = pd.read_csv(uploaded_file)
    st.sidebar.success("Data berhasil dimuat!")
else:
    st.sidebar.write("Atau gunakan data default:")
    if st.sidebar.button("Gunakan Data Default"):
        data = pd.read_csv("/path/to/your/default/case1.csv")  # Ganti path sesuai kebutuhan
        st.sidebar.success("Data default dimuat!")

# Menampilkan data jika ada
if "data" in locals():
    st.subheader("Data yang Dimuat")
    st.dataframe(data)

    # Panel Preprocessing
    st.sidebar.header("2. Preprocessing")
    if st.sidebar.button("Standarisasi Data"):
        scaler = StandardScaler()
        processed_data = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))
        st.sidebar.success("Data telah distandarisasi.")
        st.subheader("Data yang Telah Distandarisasi")
        st.dataframe(pd.DataFrame(processed_data, columns=data.select_dtypes(include=['float64', 'int64']).columns))

    # Panel Elbow Method
    st.sidebar.header("3. Elbow Method")
    if st.sidebar.button("Lihat Grafik Elbow"):
        sse = []
        k_range = range(1, 11)
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42)
            kmeans.fit(processed_data)
            sse.append(kmeans.inertia_)

        fig, ax = plt.subplots()
        ax.plot(k_range, sse, marker="o")
        ax.set_xlabel("Jumlah Klaster (k)")
        ax.set_ylabel("Sum of Squared Errors (SSE)")
        ax.set_title("Metode Elbow")
        st.pyplot(fig)

    # Panel Clustering
    st.sidebar.header("4. Clustering")
    k = st.sidebar.number_input("Jumlah Klaster (k):", min_value=2, max_value=10, value=3)
    if st.sidebar.button("Jalankan K-Means"):
        kmeans = KMeans(n_clusters=k, random_state=42)
        clustering_labels = kmeans.fit_predict(processed_data)
        data["Cluster"] = clustering_labels
        st.sidebar.success(f"Clustering selesai dengan k={k}.")
        st.subheader("Data dengan Hasil Clustering")
        st.dataframe(data)

    # Panel Evaluasi
    st.sidebar.header("5. Evaluation")
    if st.sidebar.button("Evaluasi Klaster"):
        silhouette_avg = silhouette_score(processed_data, clustering_labels)
        st.sidebar.write(f"Silhouette Score: {silhouette_avg:.2f}")

    # Visualisasi Klaster
    st.sidebar.header("6. Visualisasi Klaster")
    visualization_choice = st.sidebar.selectbox("Pilih Visualisasi", ["2D PCA", "3D PCA"])
    if st.sidebar.button("Tampilkan Visualisasi"):
        if visualization_choice == "2D PCA":
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(processed_data)
            fig, ax = plt.subplots()
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=clustering_labels, cmap="viridis", s=50)
            ax.set_title("Visualisasi Klaster 2D")
            ax.set_xlabel("Komponen Utama 1")
            ax.set_ylabel("Komponen Utama 2")
            fig.colorbar(scatter, label="Klaster")
            st.pyplot(fig)
        elif visualization_choice == "3D PCA":
            pca = PCA(n_components=3)
            reduced_data = pca.fit_transform(processed_data)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2], c=clustering_labels, cmap="viridis", s=50)
            ax.set_title("Visualisasi Klaster 3D")
            ax.set_xlabel("Komponen Utama 1")
            ax.set_ylabel("Komponen Utama 2")
            ax.set_zlabel("Komponen Utama 3")
            fig.colorbar(scatter, label="Klaster")
            st.pyplot(fig)

    # Panel Unduh Data
    st.sidebar.header("7. Unduh Hasil Clustering")
    if st.sidebar.button("Unduh Data Clustered"):
        st.markdown(download_csv(data), unsafe_allow_html=True)
