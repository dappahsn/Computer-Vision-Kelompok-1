import streamlit as st
import cv2
import numpy as np
import base64

# Fungsi untuk mengubah gambar lokal menjadi base64 (untuk background)
def get_base64_of_bin_file(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()

# Konfigurasi halaman (menggunakan ikon lokal)
st.set_page_config(
    page_title="Adaptive Sobel Edge Detection",
    layout="wide",
    page_icon="img/logo-usk.png"  # Ikon lokal
)

# Terapkan background image dari file lokal
bg_image = get_base64_of_bin_file("img/bg.jpg")
page_bg_img = f"""
<style>
.stApp {{
    background-image: url("data:image/jpg;base64,{bg_image}");
    background-size: cover;
    background-repeat: no-repeat;
    background-attachment: fixed;
}}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Styling font
st.markdown(
    """
    <link href="https://fonts.googleapis.com/css2?family=Poppins&display=swap" rel="stylesheet">
    <style>
    * {
        font-family: 'Poppins', sans-serif;
        font-weight: 500;
        font-size: 16px;
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
    st.image("img/sobel.png", use_container_width=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

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
        st.subheader("Hasil Sobel Standart")
        st.image(sobel_uint8, clamp=True, caption="Sobel Standart", use_container_width=True)

        st.subheader("Hasil Adaptive Sobel (Otsu)")
        st.image(otsu_thresh, clamp=True, caption="Adaptive Sobel (Otsu)", use_container_width=True)

    # Informasi tambahan
    st.markdown("---")
    st.image("img/logo-usk-hitam.png", width=120)

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
    st.image("img/logo-usk-hitam.png", width=120)

    st.markdown("### Computer Vision - Kelompok 1:")
    st.markdown("""
    1. Muhammad Daffa Husen  
    2. Miftah Rizki Pohan  
    3. Ryan Akmal Pasya  
    4. Muhammad Iffat Najwan  
    5. Hanum Aulia Ramadhani
    """)
