import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# Load model
model = tf.keras.models.load_model(
    "MobileNetV3Small_RealWorld_Best_v2.keras",
    compile=False
)

# Judul aplikasi
st.title("AI-Generated Image Classifier")

st.write(
    "Upload gambar untuk mengetahui apakah gambar "
    "termasuk AI_GENERATED atau HUMAN_CREATED."
)

# Upload gambar
uploaded_file = st.file_uploader(
    "Pilih gambar",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    # Buka gambar
    image = Image.open(uploaded_file).convert("RGB")

    # Tampilkan gambar
    st.image(
        image,
        caption="Gambar yang diupload",
        use_container_width=True
    )

    # Resize sesuai input MobileNetV3Small
    image_resized = image.resize((224, 224))

    # Convert ke array
    img_array = np.array(image_resized)

    # Tambahkan batch dimension
    img_array = np.expand_dims(img_array, axis=0)

    # Prediksi
    prediction = model.predict(
        img_array,
        verbose=0
    )[0][0]

    # Tentukan kelas
    if prediction >= 0.5:
        result = "HUMAN_CREATED"
        confidence = prediction * 100
    else:
        result = "AI_GENERATED"
        confidence = (1 - prediction) * 100

    # Tampilkan hasil
    st.subheader("Hasil Prediksi")

    st.success(result)

    st.write(
        f"Confidence: **{confidence:.2f}%**"
    )