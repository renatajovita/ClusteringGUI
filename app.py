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

# Session state untuk menyimpan data
if "data" not in st.session_state:
    st.session_state["data"] = None
if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = None
if "clustering_labels" not in st.session_state:
    st.session_state["clustering_labels"] = None
if "data_ready" not in st.session_state:
    st.session_state["data_ready"] = False  # Menyimpan status apakah data sudah siap untuk analisis

# Fungsi reset untuk mengembalikan session state ke keadaan awal
def reset_state():
    st.session_state["data"] = None
    st.session_state["processed_data"] = None
    st.session_state["clustering_labels"] = None
    st.session_state["data_ready"] = False

# Fungsi layout untuk tombol
def layout_buttons():
    col1, col2, col3 = st.columns([1, 3, 1])
    with col1:
        if st.button("Reset"):
            reset_state()
            st.success("Semua data dan analisis telah direset.")
    with col2:
        # Navigation bar yang digunakan untuk memilih halaman
        menu = st.radio(
            "Navigasi",
            ["Upload Data", "Preprocessing", "Elbow Method", "Clustering", "Evaluation", "Visualization", "Relabel", "Download"],
            horizontal=True
        )
    return menu

# Menampilkan tombol Reset dan navigasi
menu = layout_buttons()

# Halaman Upload Data
if menu == "Upload Data":
    st.title("1. Upload Data")
    col1, col2 = st.columns([3, 1])
    with col1:
        uploaded_file = st.file_uploader("Unggah file CSV Anda", type=["csv"])
    with col2:
        if st.button("Gunakan Data Default"):
            st.session_state["data"] = pd.read_csv("case1.csv")  # Ganti path sesuai
            st.session_state["data_ready"] = False  # Reset setelah menggunakan data default
            st.success("Data default dimuat!")

    if uploaded_file:
        st.session_state["data"] = pd.read_csv(uploaded_file)
        st.session_state["data_ready"] = False  # Reset setelah upload file baru
        st.success("Data berhasil dimuat!")

    if st.session_state["data"] is not None:
        st.write("Data yang Dimuat:")
        st.dataframe(st.session_state["data"])
    
    # Menambahkan tombol Analyze
    if st.button("Analyze"):
        st.session_state["data_ready"] = True  # Menandakan data siap untuk diproses
        st.success("Data siap untuk dianalisis! Klik tab selanjutnya untuk melanjutkan.")

# Halaman Preprocessing
elif menu == "Preprocessing":
    st.title("2. Preprocessing")
    if not st.session_state["data_ready"]:
        st.warning("Silakan klik 'Analyze' di tab 'Upload Data' untuk melanjutkan.")
    else:
        if st.button("Standarisasi Data"):
            scaler = StandardScaler()
            st.session_state["processed_data"] = scaler.fit_transform(
                st.session_state["data"].select_dtypes(include=['float64', 'int64'])
            )
            st.success("Data telah distandarisasi!")
            st.write(pd.DataFrame(
                st.session_state["processed_data"],
                columns=st.session_state["data"].select_dtypes(include=['float64', 'int64']).columns
            ))

# Halaman Elbow Method
elif menu == "Elbow Method":
    st.title("3. Elbow Method")
    if st.session_state["processed_data"] is None:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu.")
    else:
        if st.button("Lihat Grafik Elbow"):
            sse = []
            k_range = range(1, 11)
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(st.session_state["processed_data"])
                sse.append(kmeans.inertia_)
            fig, ax = plt.subplots()
            ax.plot(k_range, sse, marker="o")
            ax.set_xlabel("Jumlah Klaster (k)")
            ax.set_ylabel("Sum of Squared Errors (SSE)")
            ax.set_title("Metode Elbow")
            st.pyplot(fig)

# Halaman Clustering
elif menu == "Clustering":
    st.title("4. Clustering")
    if st.session_state["processed_data"] is None:
        st.warning("Silakan lakukan preprocessing data terlebih dahulu.")
    else:
        k = st.number_input("Jumlah Klaster (k):", min_value=2, max_value=10, value=3)
        if st.button("Jalankan K-Means"):
            kmeans = KMeans(n_clusters=k, random_state=42)
            st.session_state["clustering_labels"] = kmeans.fit_predict(st.session_state["processed_data"])
            st.session_state["data"]["Cluster"] = st.session_state["clustering_labels"]
            st.success(f"Clustering selesai dengan k={k}.")
            st.write("Data dengan hasil clustering:")
            st.dataframe(st.session_state["data"])

# Halaman Evaluation
elif menu == "Evaluation":
    st.title("5. Evaluation")
    if st.session_state["clustering_labels"] is None:
        st.warning("Silakan lakukan clustering terlebih dahulu.")
    else:
        if st.button("Evaluasi Klaster"):
            silhouette_avg = silhouette_score(
                st.session_state["processed_data"], st.session_state["clustering_labels"]
            )
            st.write(f"Silhouette Score: {silhouette_avg:.2f}")

# Halaman Visualization
elif menu == "Visualization":
    st.title("6. Visualization")
    if st.session_state["clustering_labels"] is None:
        st.warning("Silakan lakukan clustering terlebih dahulu.")
    else:
        choice = st.radio("Pilih Visualisasi", ["2D PCA", "3D PCA"])
        if st.button("Tampilkan Visualisasi"):
            if choice == "2D PCA":
                pca = PCA(n_components=2)
                reduced_data = pca.fit_transform(st.session_state["processed_data"])
                fig, ax = plt.subplots()
                scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1],
                                     c=st.session_state["clustering_labels"], cmap="viridis")
                ax.set_title("Visualisasi Klaster 2D")
                ax.set_xlabel("Komponen Utama 1")
                ax.set_ylabel("Komponen Utama 2")
                st.pyplot(fig)
            elif choice == "3D PCA":
                pca = PCA(n_components=3)
                reduced_data = pca.fit_transform(st.session_state["processed_data"])
                fig = plt.figure()
                ax = fig.add_subplot(111, projection="3d")
                scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], reduced_data[:, 2],
                                     c=st.session_state["clustering_labels"], cmap="viridis")
                ax.set_title("Visualisasi Klaster dalam 3D")
                ax.set_xlabel("Komponen Utama 1")
                ax.set_ylabel("Komponen Utama 2")
                ax.set_zlabel("Komponen Utama 3")
                st.pyplot(fig)

# Halaman Relabel Clusters
elif menu == "Relabel":
    st.title("7. Relabel Clusters")
    if st.session_state["clustering_labels"] is None:
        st.warning("Silakan lakukan clustering terlebih dahulu.")
    else:
        st.write("Masukkan label baru untuk setiap klaster:")
        unique_labels = sorted(set(st.session_state["clustering_labels"]))
        new_labels = {}
        for label in unique_labels:
            new_label = st.text_input(f"Cluster {label}", f"Cluster {label}")
            new_labels[label] = new_label
        
        if st.button("Update Labels"):
            st.session_state["data"]["Cluster"] = st.session_state["data"]["Cluster"].map(new_labels)
            st.success("Label klaster berhasil diperbarui!")
            st.write(st.session_state["data"])

# Halaman Download
elif menu == "Download":
    st.title("8. Download")
    if st.session_state["data"] is None or "Cluster" not in st.session_state["data"].columns:
        st.warning("Tidak ada data untuk diunduh.")
    else:
        st.markdown(download_csv(st.session_state["data"]), unsafe_allow_html=True)
