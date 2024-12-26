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

# Session state untuk menyimpan data
if "data" not in st.session_state:
    st.session_state["data"] = None
if "data_source" not in st.session_state:
    st.session_state["data_source"] = None  # Menyimpan sumber data ("upload" atau "default")
if "processed_data" not in st.session_state:
    st.session_state["processed_data"] = None
if "clustering_labels" not in st.session_state:
    st.session_state["clustering_labels"] = None
if "relabel_mapping" not in st.session_state:
    st.session_state["relabel_mapping"] = {}

# Tab Navigasi
tabs = st.tabs(["Kelompok", "Upload Data", "Preprocessing", "Elbow Method", "Clustering", "Evaluation", "Visualization", "Relabel Clusters", "Download"])

# Tab 1: Kelompok
with tabs[0]:
    st.title("Clustering Analysis GUI")
    st.write("""
    ## Disusun oleh:
    - **Alya Faadhila Rosyid** (24050122130049)
    - **Mesakh Besta Anugrah** (24050122130058)
    - **Renata Jovita Aurelia Dinda** (24050122130094)
    
    ### DEPARTEMEN STATISTIKA  
    FAKULTAS SAINS DAN MATEMATIKA  
    UNIVERSITAS DIPONEGORO  
    SEMARANG - 2024
    """)

# Tab 2: Upload Data
with tabs[1]:
    st.title("1. Upload Data")
    
    # Tampilkan sumber data aktif
    if st.session_state["data_source"]:
        st.info(f"**Sumber Data Saat Ini**: {st.session_state['data_source'].capitalize()}")

    # Opsi unggah data
    uploaded_file = st.file_uploader("Unggah file CSV Anda", type=["csv"])
    if uploaded_file:
        st.session_state["data"] = pd.read_csv(uploaded_file)
        st.session_state["data_source"] = "upload"
        st.success("Data berhasil diunggah dan digunakan!")
    
    # Opsi gunakan data default
    if st.button("Gunakan Data Default"):
        st.session_state["data"] = pd.read_csv("case1.csv")  # Ganti path sesuai
        st.session_state["data_source"] = "default"
        st.success("Data default berhasil digunakan!")
    
    # Reset data
    if st.button("Reset Data"):
        st.session_state["data"] = None
        st.session_state["data_source"] = None
        st.session_state["processed_data"] = None
        st.session_state["clustering_labels"] = None
        st.session_state["relabel_mapping"] = {}
        st.warning("Data telah direset.")

    # Tampilkan data saat ini
    if st.session_state["data"] is not None:
        st.write("**Data yang Dimuat Saat Ini:**")
        st.dataframe(st.session_state["data"])
    else:
        st.warning("Belum ada data yang dimuat. Silakan unggah data atau gunakan data default.")

# Tab 3: Preprocessing
with tabs[2]:
    st.title("2. Preprocessing")
    if st.session_state["data"] is None:
        st.warning("Silakan unggah data terlebih dahulu di tab 'Upload Data'.")
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

# Tab 4: Elbow Method
with tabs[3]:
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

# Tab 5: Clustering
with tabs[4]:
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

# Tab 6: Evaluation
with tabs[5]:
    st.title("5. Evaluation")
    if st.session_state["clustering_labels"] is None:
        st.warning("Silakan lakukan clustering terlebih dahulu.")
    else:
        if st.button("Evaluasi Klaster"):
            silhouette_avg = silhouette_score(
                st.session_state["processed_data"], st.session_state["clustering_labels"]
            )
            st.write(f"Silhouette Score: {silhouette_avg:.2f}")

# Tab 7: Visualization
with tabs[6]:
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
                ax.set_title("Visualisasi Klaster 3D")
                ax.set_xlabel("Komponen Utama 1")
                ax.set_ylabel("Komponen Utama 2")
                ax.set_zlabel("Komponen Utama 3")
                st.pyplot(fig)

# Tab 8: Relabel Clusters
with tabs[7]:
    st.title("7. Relabel Clusters")
    if st.session_state["clustering_labels"] is None:
        st.warning("Silakan lakukan clustering terlebih dahulu.")
    else:
        unique_clusters = sorted(set(st.session_state["clustering_labels"]))
        for cluster in unique_clusters:
            new_label = st.text_input(f"Label baru untuk Cluster {cluster}:", f"Cluster {cluster}")
            st.session_state["relabel_mapping"][cluster] = new_label

        if st.button("Perbarui Label"):
            st.session_state["data"]["Cluster"] = st.session_state["data"]["Cluster"].map(st.session_state["relabel_mapping"])
            st.success("Label berhasil diperbarui!")
            st.write("Data dengan label baru:")
            st.dataframe(st.session_state["data"])

# Tab 9: Download
with tabs[8]:
    st.title("8. Download")
    if st.session_state["data"] is None or "Cluster" not in st.session_state["data"].columns:
        st.warning("Tidak ada data untuk diunduh.")
    else:
        st.markdown(download_csv(st.session_state["data"]), unsafe_allow_html=True)
