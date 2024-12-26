import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Fungsi untuk menampilkan data dalam tabel
def display_data(data, title="Data Preview"):
    st.write(title)
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

    # Inisialisasi
    if "data" not in st.session_state:
        st.session_state.data = None
        st.session_state.processed_data = None
        st.session_state.labels = None

    # Tab panel
    tabs = st.tabs(["1. Upload Data", "2. Preprocessing & Elbow Method", "3. Clustering", "4. Evaluation", "5. Relabel & Export"])
    
    # Tab 1: Upload Data
    with tabs[0]:
        uploaded_file = st.file_uploader("Upload CSV File", type="csv")
        use_default = st.checkbox("Gunakan Data Default")
        
        if uploaded_file:
            st.session_state.data = pd.read_csv(uploaded_file)
            st.success("Data berhasil dimuat dari file upload.")
        elif use_default:
            default_path = "case1.csv"
            try:
                st.session_state.data = pd.read_csv(default_path)
                st.success("Data default berhasil dimuat.")
            except FileNotFoundError:
                st.error("File default tidak ditemukan. Pastikan file 'case1.csv' ada di repositori.")
        
        if st.session_state.data is not None:
            display_data(st.session_state.data, "Data Asli")

    # Tab 2: Preprocessing & Elbow Method
    with tabs[1]:
        if st.session_state.data is not None:
            numeric_columns = st.session_state.data.select_dtypes(include=["float64", "int64"]).columns
            scaler = StandardScaler()
            st.session_state.processed_data = scaler.fit_transform(st.session_state.data[numeric_columns])
            processed_df = pd.DataFrame(st.session_state.processed_data, columns=numeric_columns)
            display_data(processed_df, "Data Distandarisasi")
            
            st.write("### Elbow Method")
            sse = []
            k_range = range(1, 11)
            for k in k_range:
                kmeans = KMeans(n_clusters=k, random_state=42)
                kmeans.fit(st.session_state.processed_data)
                sse.append(kmeans.inertia_)
            
            fig, ax = plt.subplots()
            ax.plot(k_range, sse, marker='o')
            ax.set_title("Elbow Method")
            ax.set_xlabel("Jumlah Klaster (k)")
            ax.set_ylabel("Sum of Squared Errors (SSE)")
            st.pyplot(fig)
        else:
            st.warning("Silakan upload data terlebih dahulu.")

    # Tab 3: Clustering
    with tabs[2]:
        if st.session_state.processed_data is not None:
            k = st.slider("Pilih jumlah klaster (k):", min_value=2, max_value=10, value=3)
            if st.button("Jalankan Clustering"):
                kmeans = KMeans(n_clusters=k, random_state=42)
                st.session_state.labels = kmeans.fit_predict(st.session_state.processed_data)
                st.session_state.data["Cluster"] = st.session_state.labels
                st.success("Clustering selesai!")
                display_data(st.session_state.data, "Data dengan Klaster")
        else:
            st.warning("Silakan lakukan preprocessing terlebih dahulu.")

    # Tab 4: Evaluation
    with tabs[3]:
        if st.session_state.labels is not None:
            silhouette_avg = silhouette_score(st.session_state.processed_data, st.session_state.labels)
            st.write(f"### Silhouette Score: {silhouette_avg:.2f}")

            # Visualisasi Klaster 2D
            st.write("### Visualisasi Klaster 2D")
            pca = PCA(n_components=2)
            reduced_data = pca.fit_transform(st.session_state.processed_data)
            fig, ax = plt.subplots()
            scatter = ax.scatter(reduced_data[:, 0], reduced_data[:, 1], c=st.session_state.labels, cmap="viridis", s=50)
            ax.set_title("Klaster 2D")
            ax.set_xlabel("Komponen Utama 1")
            ax.set_ylabel("Komponen Utama 2")
            fig.colorbar(scatter, label="Klaster")
            st.pyplot(fig)

            # Visualisasi Klaster 3D
            st.write("### Visualisasi Klaster 3D")
            pca_3d = PCA(n_components=3)
            reduced_data_3d = pca_3d.fit_transform(st.session_state.processed_data)
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            scatter = ax.scatter(reduced_data_3d[:, 0], reduced_data_3d[:, 1], reduced_data_3d[:, 2], 
                                  c=st.session_state.labels, cmap="viridis", s=50)
            ax.set_title("Klaster 3D")
            ax.set_xlabel("Komponen Utama 1")
            ax.set_ylabel("Komponen Utama 2")
            ax.set_zlabel("Komponen Utama 3")
            fig.colorbar(scatter, label="Klaster")
            st.pyplot(fig)
        else:
            st.warning("Lakukan clustering terlebih dahulu.")

    # Tab 5: Relabel & Export
    with tabs[4]:
        if "Cluster" in st.session_state.data.columns:
            st.write("### Relabel Clusters")
            unique_clusters = sorted(st.session_state.data["Cluster"].unique())
            new_labels = {}
            for cluster in unique_clusters:
                new_label = st.text_input(f"Label untuk Cluster {cluster}:", f"Cluster {cluster}")
                new_labels[cluster] = new_label

            if st.button("Update Labels"):
                st.session_state.data["Cluster"] = st.session_state.data["Cluster"].map(new_labels)
                st.success("Label klaster berhasil diperbarui!")
                display_data(st.session_state.data, "Data dengan Label Baru")

            # Export Data
            st.write("### Export Data")
            csv = st.session_state.data.to_csv(index=False)
            st.download_button(
                label="Download Data Clustered (CSV)",
                data=csv,
                file_name="clustered_data.csv",
                mime="text/csv"
            )
        else:
            st.warning("Lakukan clustering terlebih dahulu.")

if __name__ == "__main__":
    main()
