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

# Variabel global
data = None
processed_data = None
clustering_labels = None

# Sidebar Navigasi
st.sidebar.title("Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["Upload Data", "Preprocessing", "Elbow Method", "Clustering", "Evaluation", "Visualization", "Download"]
)

# Fungsi untuk memuat data
def load_data():
    uploaded_file = st.sidebar.file_uploader("Unggah file CSV Anda", type=["csv"])
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file)
    elif st.sidebar.button("Gunakan Data Default"):
        # Ganti path ini dengan data default Anda
        return pd.read_csv("case1.csv")
    return None

# Halaman Upload Data
if page == "Upload Data":
    st.subheader("1. Upload Data")
    data = load_data()
    if data is not None:
        st.write("Data yang dimuat:")
        st.dataframe(data)
    else:
        st.warning("Unggah file CSV terlebih dahulu atau gunakan data default.")

# Halaman Preprocessing
if page == "Preprocessing":
    if data is None:
        st.warning("Silakan unggah data terlebih dahulu di halaman 'Upload Data'.")
    else:
        st.subheader("2. Preprocessing")
        if st.button("Standarisasi Data"):
            scaler = StandardScaler()
            processed_data = scaler.fit_transform(data.select_dtypes(include=['float64', 'int64']))
            st.success("Data telah distandarisasi.")
            st.write("Data yang telah distandarisasi:")
            st.dataframe(pd.DataFrame(processed_data, columns=data.select_dtypes(include=['float64', 'int64']).columns))

# Halaman Elbow Method
if page == "Elbow Method":
    if processed_data is None:
        st.warning("Silakan lakukan preprocessing data di halaman 'Preprocessing'.")
    else:
        st.subheader("3. Elbow Method")
        if st.button("Lihat Grafik Elbow"):
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

# Halaman Clustering
if page == "Clustering":
    if processed_data is None:
        st.warning("Silakan lakukan preprocessing data di halaman 'Preprocessing'.")
    else:
        st.subheader("4. Clustering")
        k = st.number_input("Jumlah Klaster (k):", min_value=2, max_value=10, value=3)
        if st.button("Jalankan K-Means"):
            kmeans = KMeans(n_clusters=k, random_state=42)
            clustering_labels = kmeans.fit_predict(processed_data)
            data["Cluster"] = clustering_labels
            st.success(f"Clustering selesai dengan k={k}.")
            st.write("Data dengan hasil clustering:")
            st.dataframe(data)

# Halaman Evaluation
if page == "Evaluation":
    if clustering_labels is None:
        st.warning("Silakan lakukan clustering di halaman 'Clustering'.")
    else:
        st.subheader("5. Evaluation")
        if st.button("Evaluasi Klaster"):
            silhouette_avg = silhouette_score(processed_data, clustering_labels)
            st.write(f"Silhouette Score: {silhouette_avg:.2f}")

# Halaman Visualization
if page == "Visualization":
    if clustering_labels is None:
        st.warning("Silakan lakukan clustering di halaman 'Clustering'.")
    else:
        st.subheader("6. Visualization")
        visualization_choice = st.selectbox("Pilih Visualisasi", ["2D PCA", "3D PCA"])
        if st.button("Tampilkan Visualisasi"):
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

# Halaman Download
if page == "Download":
    if data is None or "Cluster" not in data.columns:
        st.warning("Silakan lakukan clustering di halaman 'Clustering'.")
    else:
        st.subheader("7. Download")
        st.markdown(download_csv(data), unsafe_allow_html=True)
