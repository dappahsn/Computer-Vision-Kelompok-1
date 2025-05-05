import streamlit as st
import cv2
import numpy as np
from PIL import Image
from streamlit_option_menu import option_menu

# Judul aplikasi
st.title('Edge Detecting Using Adaptive Sobel')

# Upload gambar
uploaded_file = st.file_uploader("Upload gambar", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Membaca gambar
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)  # 1 artinya baca gambar berwarna
    
    # Konversi ke grayscale
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Sobel
    sobelx = cv2.Sobel(img_gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(img_gray, cv2.CV_64F, 0, 1, ksize=3)
    sobel_magnitude = cv2.magnitude(sobelx, sobely)

    # Otsu Threshold
    sobel_uint8 = np.uint8(sobel_magnitude)
    _, otsu_thresh = cv2.threshold(sobel_uint8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Tampilkan gambar
    st.subheader('Gambar Asli')
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), channels="RGB")

    st.subheader('Gambar GreyScale')
    st.image(img_gray, clamp=True)

    st.subheader('Hasil Sobel Standart')
    st.image(sobel_magnitude, clamp=True)

    st.subheader('Hasil Adaptive Sobel Otsu Threshold')
    st.image(otsu_thresh, clamp=True)
