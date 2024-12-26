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
    b64 = base64.b64encode(csv.encode()).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{filename}">Klik di sini untuk mengunduh file CSV</a>'
    return href

# Konfigurasi Streamlit
st.set_page_config(page_title="Clustering Analysis App", layout="wide")

# Inisialisasi session state
if "data" not in st.session_state:
    st.session_state["data"] = None
if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = None
if "clustering_labels" not in st.session_state:
    st.session_state["clustering_labels"] = None
if "analyzed" not in st.session_state:
    st.session_state["analyzed"] = False

# Tab Navigasi
tabs = st.tabs(["Upload Data", "Preprocessing", "Elbow Method", "Clustering", "Visualization", "Download"])

# Tab 1: Upload Data
with tabs[0]:
    st.title("1. Upload Data")

    # Input data
    uploaded_file = st.file_uploader("Unggah file CSV Anda", type=["csv"])
    col1, col2 = st.columns(2)
    with col1:
        use_default = st.button("Gunakan Data Default")
    with col2:
        analyze_button = st.button("Analyze")

    # Reset data jika file baru diunggah atau default dipilih
    if uploaded_file:
        st.session_state["data"] = pd.read_csv(uploaded_file)
        st.session_state["processed_data"] = None
        st.session_state["clustering_labels"] = None
        st.session_state["analyzed"] = False
        st.success("Data berhasil dimuat dari file.")
    elif use_default:
        st.session_state["data"] = pd.read_csv("case1.csv")  # Ganti path sesuai
        st.session_state["processed_data"] = None
        st.session_state["clustering_labels"] = None
        st.session_state["analyzed"] = False
        st.success("Data default berhasil dimuat.")

    # Tombol Analyze
    if analyze_button:
        if st.session_state["data"] is not None:
            st.session_state["analyzed"] = True
            st.success("Data siap untuk analisis selanjutnya.")
        else:
            st.error("Silakan unggah data atau gunakan data default sebelum melanjutkan.")

    # Tampilkan data yang dimuat
    if st.session_state["data"] is not None:
        st.write("**Data yang Dimuat:**")
        st.dataframe(st.session_state["data"])

# Tab 2: Preprocessing
with tabs[1]:
    st.title("2. Preprocessing")
    if not st.session_state["analyzed"]:
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

# Tab 3: Elbow Method
with tabs[2]:
    st.title("3. Elbow Method")
    if not st.session_state["analyzed"] or st.session_state["processed_data"] is None:
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

# Tab 4: Clustering
with tabs[3]:
    st.title("4. Clustering")
    if not st.session_state["analyzed"] or st.session_state["processed_data"] is None:
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

# Tab 5: Visualization
with tabs[4]:
    st.title("5. Visualization")
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
                ax.set_title("Visualisasi Klaster 3D")
                ax.set_xlabel("Komponen Utama 1")
                ax.set_ylabel("Komponen Utama 2")
                ax.set_zlabel("Komponen Utama 3")
                st.pyplot(fig)

# Tab 6: Download
with tabs[5]:
    st.title("6. Download")
    if st.session_state["data"] is None or "Cluster" not in st.session_state["data"].columns:
        st.warning("Tidak ada data untuk diunduh.")
    else:
        st.markdown(download_csv(st.session_state["data"]), unsafe_allow_html=True)
