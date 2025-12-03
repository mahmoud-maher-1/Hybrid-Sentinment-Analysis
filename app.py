import streamlit as st
import tensorflow as tf
import pickle
import os
import re
import string
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

# Page Config
st.set_page_config(page_title="Sentiment Analysis Suite", layout="wide")

# Constants
MODEL_DIR = "Models"
MAX_LEN = 200


# Helper Functions (Replicating logic to avoid importing from src in deployment scenarios)
def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'http\S+|www\.\S+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = re.sub(r'\d+', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


@st.cache_resource
def load_resources(model_name):
    # Load Tokenizer
    with open(os.path.join(MODEL_DIR, "tokenizer.pickle"), 'rb') as handle:
        tokenizer = pickle.load(handle)

    # Load Model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.h5")
    model = tf.keras.models.load_model(model_path)
    return tokenizer, model


# Sidebar
st.sidebar.title("Configuration")
model_choice = st.sidebar.radio(
    "Select Model Architecture",
    ("lstm", "bi-lstm", "gru", "bi-gru")
)

st.sidebar.markdown("---")
st.sidebar.info("Models were trained on the IMDB dataset using Keras.")

# Main Interface
st.title("🎬 Movie Review Sentiment Analysis")
st.markdown("Enter a review below to detect if the sentiment is **Positive** or **Negative**.")

user_input = st.text_area("Review Text", height=150, placeholder="Type your review here...")

if st.button("Analyze Sentiment"):
    if user_input.strip() == "":
        st.warning("Please enter some text.")
    else:
        try:
            tokenizer, model = load_resources(model_choice)

            # Preprocess
            cleaned = clean_text(user_input)
            seq = tokenizer.texts_to_sequences([cleaned])
            padded = pad_sequences(seq, maxlen=MAX_LEN, padding='post', truncating='post')

            # Predict
            pred_prob = model.predict(padded)[0][0]
            sentiment = "Positive" if pred_prob > 0.5 else "Negative"
            confidence = pred_prob if pred_prob > 0.5 else 1 - pred_prob

            # Display
            st.markdown("### Results")
            col1, col2 = st.columns(2)

            with col1:
                color = "green" if sentiment == "Positive" else "red"
                st.markdown(
                    f"Sentiment: <span style='color:{color}; font-size:24px; font-weight:bold'>{sentiment}</span>",
                    unsafe_allow_html=True)

            with col2:
                st.metric("Confidence", f"{confidence:.2%}")
                st.progress(float(confidence))

        except Exception as e:
            st.error(f"Error loading model resources. Ensure 'src/main.py' has been run first.\nDetails: {e}")

# Performance Section
st.markdown("---")
st.subheader("Model Performance")
try:
    st.image(f"Visualizations/{model_choice}_cm.png", caption=f"{model_choice.upper()} Confusion Matrix")
except:
    st.info("Run the training pipeline to generate visualizations.")