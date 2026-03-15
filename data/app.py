import streamlit as st
import joblib
from fer import FER
import cv2
import numpy as np
import librosa
import tempfile
import os

# Load typing model
if not os.path.exists("model/stress_model.pkl"):
    st.error("Model not found! Run train_model.py first.")
    st.stop()
typing_model = joblib.load("model/stress_model.pkl")

st.title("Real AI Stress Detection App")
st.write("Detect stress using typing, face, and voice inputs")

# ---------- TYPING ----------
st.subheader("Typing Behavior")
typing_speed = st.slider("Typing Speed (wpm)", 20, 100, 50)
typing_errors = st.slider("Typing Errors", 0, 10, 2)
screen_time = st.slider("Screen Time (hours/day)", 0, 10, 3)

typing_pred = typing_model.predict([[typing_speed, typing_errors, screen_time]])[0]

# ---------- FACE ----------
st.subheader("Face / Emotion Detection")
camera_image = st.camera_input("Take a selfie")

face_emotion = "N/A"
if camera_image:
    image_bytes = np.asarray(bytearray(camera_image.read()), dtype=np.uint8)
    img = cv2.imdecode(image_bytes, cv2.IMREAD_COLOR)
    detector = FER(mtcnn=True)
    result = detector.detect_emotions(img)
    if result:
        face_emotion = max(result[0]['emotions'], key=result[0]['emotions'].get)
st.write(f"Detected Face Emotion: {face_emotion}")

# ---------- VOICE ----------
st.subheader("Voice Stress Detection")
audio_file = st.file_uploader("Upload a 3-second voice clip", type=["wav", "mp3"])
voice_emotion = "N/A"
if audio_file:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(audio_file.read())
        y, sr = librosa.load(tmp_file.name, duration=3)
        mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
        avg_mfcc = np.mean(mfccs)
        if avg_mfcc < -200:
            voice_emotion = "Relaxed"
        elif avg_mfcc < -150:
            voice_emotion = "Moderate"
        else:
            voice_emotion = "High"
st.write(f"Detected Voice Emotion: {voice_emotion}")

# ---------- FINAL STRESS ----------
st.subheader("Final Stress Prediction")
if typing_pred == 0:
    st.success("Relaxed 😊")
elif typing_pred == 1:
    st.warning("Moderate Stress 😐")
else:
    st.error("High Stress 😣")
