import streamlit as st
import numpy as np
import joblib

st.set_page_config(page_title="Music Genre Predictor", layout="centered")

st.title("🎵 Music Genre Predictor")
st.write("Use the sliders to predict a song's genre.")

model = joblib.load("genre_knn.pkl")

tempo = st.slider("Tempo (BPM)", 50, 200, 120)
energy = st.slider("Energy", 0.0, 1.0, 0.5)
danceability = st.slider("Danceability", 0.0, 1.0, 0.5)
acousticness = st.slider("Acousticness", 0.0, 1.0, 0.5)

if st.button("Predict Genre"):
    X = np.array([[tempo, energy, danceability, acousticness]])

    prediction = model.predict(X)[0]

    st.success(f"Predicted Genre: {prediction}")

    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X)[0]
        classes = model.classes_

        st.write("### Probabilities")

        for genre, prob in zip(classes, probs):
            st.write(f"{genre}: {prob:.2f}")
            st.progress(float(prob))
        