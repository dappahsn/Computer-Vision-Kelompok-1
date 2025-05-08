import streamlit as st
import cv2
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Adaptive Sobel Edge Detection", layout="wide", page_icon="🧠")

# Styling: background, font, dan sidebar
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <style>
    * {
        font-family: 'Poppins', sans-serif;
    }
    .stApp {
        background-image: url("https://img.freepik.com/free-photo/white-paper-texture_1194-2301.jpg?t=st=1746725280~exp=1746728880~hmac=cf3277982c2e24ddedef8eb1212cb3c9b59a1694a16d19f50cf254aaf7b4662c&w=996");
        background-size: cover;
        background-repeat: no-repeat;
        background-attachment: fixed;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Judul Aplikasi
st.title("Edge Detecting Using Adaptive Sobel")
st.markdown("<br>", unsafe_allow_html=True)

# Header Penjelasan dan Gambar
col_left, col_right = st.columns([1, 1])

with col_left:
    st.markdown("""
    <div style='font-size:18px; line-height:1.6'>
    Adaptive Sobel adalah pengembangan dari metode Sobel klasik yang digunakan dalam deteksi tepi pada citra digital, dengan kemampuan untuk menyesuaikan ambang batas (threshold) secara dinamis berdasarkan karakteristik lokal citra, seperti intensitas atau kontras di sekitar piksel. Tidak seperti metode Sobel standar yang menggunakan filter tetap dan ambang batas global, pendekatan adaptif ini meningkatkan akurasi deteksi tepi, terutama pada gambar dengan pencahayaan tidak merata atau kontras rendah. Dengan memanfaatkan informasi lokal, Adaptive Sobel mampu mempertahankan detail penting dan mengurangi noise, menjadikannya lebih efektif untuk segmentasi atau analisis citra lanjutan.
    </div>
    """, unsafe_allow_html=True)

with col_right:
    st.image("sobel.png", use_container_width=True)

# Sidebar: Upload dan Pengaturan
st.sidebar.header("📥 Input & Pengaturan")

uploaded_file = st.sidebar.file_uploader("Upload gambar (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

if uploaded_file:
    st.sidebar.subheader("🛠️ Pengaturan Threshold Manual")

    threshold_manual = st.sidebar.slider(
        "Pilih nilai threshold",
        min_value=0,
        max_value=255,
        value=100,
        step=1,
        help="Atur nilai threshold manual untuk deteksi tepi"
    )

    apply_manual = st.sidebar.button("🔍 Terapkan Threshold Manual")

    # Proses gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)

    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobelx, sobely)

    sobel_norm = cv2.normalize(sobel_magnitude, None, 0, 255, cv2.NORM_MINMAX)
    sobel_uint8 = np.uint8(sobel_norm)

    _, otsu_thresh = cv2.threshold(sobel_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    thresh_manual = None
    if apply_manual:
        _, thresh_manual = cv2.threshold(sobel_uint8, threshold_manual, 255, cv2.THRESH_BINARY)

    # Tampilkan hasil
    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Gambar Asli")
        st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), caption="Gambar Asli", use_container_width=True)

        st.subheader(f"Threshold Manual (Nilai: {threshold_manual})")
        if apply_manual and thresh_manual is not None:
            st.image(thresh_manual, clamp=True, caption="Hasil Threshold Manual", use_container_width=True)
        else:
            st.info("Klik tombol *Terapkan Threshold Manual* di sidebar untuk melihat hasil.")

    with col2:
        st.subheader("Hasil Sobel Standar")
        st.image(sobel_uint8, clamp=True, caption="Sobel Magnitude", use_container_width=True)

        st.subheader("Threshold Otsu (Adaptif)")
        st.image(otsu_thresh, clamp=True, caption="Otsu Adaptive Threshold", use_container_width=True)

    # Informasi tambahan
    st.markdown("---")
    st.image("logo-usk.png", width=120)

    st.markdown("### Computer Vision - Kelompok 1:")
    st.markdown("""
    1. Muhammad Daffa Husen  
    2. Miftah Rizki Pohan  
    3. Ryan Akmal Pasya  
    4. Muhammad Iffat Najwan  
    5. Hanum Aulia Ramadhani
    """)
else:
    st.info("📂 Silakan upload gambar pada panel sidebar untuk memulai deteksi tepi.")
        # Informasi tambahan
    st.markdown("---")
    st.image("logo-usk.png", width=120)

    st.markdown("### Computer Vision - Kelompok 1:")
    st.markdown("""
    1. Muhammad Daffa Husen  
    2. Miftah Rizki Pohan  
    3. Ryan Akmal Pasya  
    4. Muhammad Iffat Najwan  
    5. Hanum Aulia Ramadhani
    """)
