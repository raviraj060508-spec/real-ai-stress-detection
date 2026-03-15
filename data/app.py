import streamlit as st
import pandas as pd
import numpy as np
import joblib
import cv2
from fer import FER
import librosa

# Load pre-trained model
model = joblib.load("model/stress_model.pkl")

st.title("🧠 Real-Time Stress Detection App")
st.write("Upload face images, audio, or CSV to detect stress levels.")

# --- Face Stress Detection ---
st.header("1️⃣ Face Stress Detection")
image_file = st.file_uploader("Upload an image", type=["jpg", "png", "jpeg"])
if image_file:
    file_bytes = np.asarray(bytearray(image_file.read()), dtype=np.uint8)
    img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    detector = FER(mtcnn=True)
    result = detector.detect_emotions(img)
    if result:
        st.image(img, channels="BGR")
        st.write("Detected emotions:", result)
    else:
        st.write("No face detected!")

# --- Audio Stress Detection ---
st.header("2️⃣ Voice Stress Detection")
audio_file = st.file_uploader("Upload a voice file", type=["wav", "mp3"])
if audio_file:
    y, sr = librosa.load(audio_file, sr=None)
    features = np.array([np.mean(y), np.std(y)]).reshape(1, -1)
    prediction = model.predict(features)
    st.write("Predicted stress level:", prediction[0])

# --- CSV Features Prediction ---
st.header("3️⃣ Predict from CSV Features")
csv_file = st.file_uploader("Upload CSV with features", type=["csv"])
if csv_file:
    df_input = pd.read_csv(csv_file)
    pred = model.predict(df_input)
    st.write("Predictions:", pred)
