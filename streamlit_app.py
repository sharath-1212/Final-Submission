import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.preprocessing import image
from PIL import Image
import os
import time
import csv
from datetime import datetime

# --- App Config ---
st.set_page_config(page_title="E-Waste Classification", layout="centered", page_icon="♻️")

# --- Sidebar ---
st.sidebar.title("♻️ E-Waste Classifier")
st.sidebar.markdown("""
This app uses a deep learning model (EfficientNetV2B3) to classify images of e-waste into 10 categories:

- Battery
- Keyboard
- Microwave
- Mobile
- Mouse
- PCB
- Player
- Printer
- Television
- Washing Machine

Upload an image to get started!
""")
st.sidebar.info("Developed by Sharath Adepu | Powered by Streamlit & TensorFlow")

# --- Main Title ---
st.markdown("""
<h1 style='text-align: center; color: #388e3c;'>E-Waste Generation Classifier 🌱</h1>
""", unsafe_allow_html=True)

st.markdown("""
<div style='text-align: center;'>
    <span style='font-size: 1.2em;'>Upload an image of e-waste and let AI classify it for you!</span>
</div>
""", unsafe_allow_html=True)

# --- Load Model and Class Labels ---
@st.cache_resource()
def load_efficientnet_model():
    return load_model("Efficient_classify.keras")

model = load_efficientnet_model()
class_labels = ['Battery', 'Keyboard', 'Microwave', 'Mobile', 'Mouse', 'PCB', 'Player', 'Printer', 'Television', 'Washing Machine']

# --- File Uploader and Layout ---
col1, col2 = st.columns([1, 2])

with col1:
    uploaded_file = st.file_uploader("Upload an E-Waste Image", type=["jpg", "jpeg", "png"], label_visibility="visible")

with col2:
    st.markdown("""
    <div style='margin-top: 1.5em; color: #616161;'>
        <b>Accepted formats:</b> JPG, JPEG, PNG<br>
        <b>Tip:</b> Use clear, well-lit images for best results.
    </div>
    """, unsafe_allow_html=True)

# --- Prediction and Display ---
if uploaded_file is not None:
    image_pil = Image.open(uploaded_file).convert("RGB")
    st.image(image_pil, caption='Uploaded Image', use_container_width=True)

    resized_image = image_pil.resize((300, 300))
    img_array = np.array(resized_image, dtype=np.float32)
    img_array = preprocess_input(img_array)
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = class_labels[np.argmax(prediction)]
    confidence = np.max(prediction) * 100

    st.markdown("---")
    st.markdown(f"<h2 style='color: #1976d2;'>🔍 Predicted Class: <span style='color: #388e3c;'>{predicted_class}</span></h2>", unsafe_allow_html=True)
    st.markdown(f"<h3 style='color: #616161;'>📊 Confidence: <span style='color: #ffa000;'>{confidence:.2f}%</span></h3>", unsafe_allow_html=True)

    # Progress bar for confidence
    st.progress(int(confidence))

    # Info box for class description (optional, can be expanded)
    class_descriptions = {
        'Battery': 'Used batteries from electronics.',
        'Keyboard': 'Computer keyboards.',
        'Microwave': 'Household microwave ovens.',
        'Mobile': 'Mobile phones and smartphones.',
        'Mouse': 'Computer mice.',
        'PCB': 'Printed Circuit Boards.',
        'Player': 'Media players (e.g., MP3, DVD).',
        'Printer': 'Printers and related devices.',
        'Television': 'TV sets and displays.',
        'Washing Machine': 'Household washing machines.'
    }
    st.info(f"**About this class:** {class_descriptions.get(predicted_class, 'No description available.')}")

    # --- Flag Feature ---
    if st.button('🚩 Flag this image'):
        base_dir = '.streamlit'
        flagged_dir = os.path.join(base_dir, 'flagged_images')
        os.makedirs(flagged_dir, exist_ok=True)
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        filename = f"flagged_{predicted_class}_{int(time.time())}.png"
        save_path = os.path.join(flagged_dir, filename)
        image_pil.save(save_path)
        st.success(f"Image has been flagged and saved as {filename}!")

        # --- Save info to dataset.csv ---
        csv_path = os.path.join(base_dir, 'dataset.csv')
        csv_exists = os.path.isfile(csv_path)
        output_str = f"Predicted: {predicted_class} (Confidence: {confidence/100:.2f})"
        with open(csv_path, 'a', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            if not csv_exists:
                writer.writerow(['img', 'output', 'timestamp'])
            writer.writerow([save_path, output_str, timestamp])

else:
    st.warning("Please upload an image to classify.")

# --- Footer ---
st.markdown("""
<hr style='border: 1px solid #e0e0e0;'>
<div style='text-align: center; color: #888;'>
    <small>© 2024 E-Waste Classifier | For educational use only</small>
</div>
""", unsafe_allow_html=True) 