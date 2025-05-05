import streamlit as st
import cv2
import numpy as np

# Konfigurasi halaman harus dipanggil di awal
st.set_page_config(page_title="Adaptive Sobel Edge Detection", layout="wide")

# Menambahkan CSS untuk desain
st.markdown(
    """
    <style>
    /* Latar belakang aplikasi */
    body {
        background-color: #f0f0f5;  /* Warna latar belakang */
        font-family: 'Arial', sans-serif;
        padding: 20px;
    }

    /* Styling untuk bagian header */
    .streamlit-expanderHeader {
        color: #005b96;  /* Warna teks header */
    }

    /* Styling untuk sidebar */
    .sidebar .sidebar-content {
        background-color: #1e1e2f;  /* Warna latar belakang sidebar */
        color: white;
    }

    .sidebar .sidebar-header {
        color: #ff6347;
    }

    /* Styling untuk bagian tombol di sidebar */
    .stButton>button {
        background-color: #0077b6; /* Warna tombol */
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-size: 14px;
        width: 100%;
    }

    /* Styling untuk gambar */
    img {
        border-radius: 10px;
    }

    /* Styling untuk subheader */
    h2, h3 {
        color: #005b96;
    }

    /* Gaya untuk bagian gambar */
    .stImage {
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
    }
    </style>
    """, unsafe_allow_html=True
)

# Judul Aplikasi
st.title("🖼️ Edge Detecting Using Adaptive Sobel")

# Sidebar: Upload dan Pengaturan
st.sidebar.header("📥 Input & Pengaturan")

uploaded_file = st.sidebar.file_uploader("Upload gambar (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# Tampilkan pengaturan hanya jika ada file yang di-upload
if uploaded_file:
    st.sidebar.subheader("🛠️ Pengaturan Threshold Manual")
    
    # Slider threshold
    threshold_manual = st.sidebar.slider(
        "Pilih nilai threshold",
        min_value=0,
        max_value=255,
        value=100,
        step=1,
        help="Atur nilai threshold manual untuk deteksi tepi"
    )

    # Tombol aplikasi threshold manual
    apply_manual = st.sidebar.button("🔍 Terapkan Threshold Manual")

    # Proses gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Sobel deteksi tepi
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobelx, sobely)

    # Normalisasi Sobel
    sobel_norm = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    sobel_uint8 = np.uint8(sobel_norm)

    # Threshold Otsu (otomatis)
    _, otsu_thresh = cv2.threshold(sobel_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Threshold manual (jika tombol ditekan)
    thresh_manual = None
    if apply_manual:
        _, thresh_manual = cv2.threshold(sobel_uint8, threshold_manual, 255, cv2.THRESH_BINARY)

    # Tampilkan hasil
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📷 Gambar Asli")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Gambar Asli", use_container_width=True)

        st.subheader(f"🎚️ Threshold Manual (Nilai: {threshold_manual})")
        if apply_manual and thresh_manual is not None:
            st.image(thresh_manual, clamp=True, caption="Hasil Threshold Manual", use_container_width=True)
        else:
            st.info("Klik tombol *Terapkan Threshold Manual* di sidebar untuk melihat hasil.")

    with col2:
        st.subheader("🔍 Hasil Sobel Standar")
        st.image(sobel_uint8, clamp=True, caption="Sobel Magnitude", use_container_width=True)

        st.subheader("⚙️ Threshold Otsu (Adaptif)")
        st.image(otsu_thresh, clamp=True, caption="Otsu Adaptive Threshold", use_container_width=True)

    # Informasi penggunaan
    st.markdown("---")
    st.markdown("### ℹ️ Cara Menggunakan Aplikasi:")
    st.markdown(""" 
    1. Upload gambar pada panel **sidebar**.
    2. Atur nilai **threshold manual** dengan slider.
    3. Klik **Terapkan Threshold Manual** untuk melihat hasilnya.
    4. Bandingkan hasil deteksi tepi antara **Otsu (otomatis)** dan **manual**.
    5. Klik kanan pada gambar → *Open image in new tab* untuk melihat lebih detail.
    """)
else:
    st.info("📂 Silakan upload gambar pada panel sidebar untuk memulai deteksi tepi.")
