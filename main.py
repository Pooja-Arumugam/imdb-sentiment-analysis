import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
import streamlit as st

# Load word index and reverse mapping
word_index = imdb.get_word_index()
reversed_word_index = {value: key for key, value in word_index.items()}

# Load the pretrained model
model = load_model('simple_rnn_imdb.h5')

# Helper functions
def decode_review(encoded_review):
    return ' '.join([reversed_word_index.get(i - 3, '?') for i in encoded_review])

def preprocess_text(text):
    words = text.lower().split()
    encoded_reviews = [word_index.get(word, 2) + 3 for word in words]
    padded_sequence = sequence.pad_sequences([encoded_reviews], maxlen=500)
    return padded_sequence

# Streamlit App
st.set_page_config(page_title="IMDB Sentiment Analyzer", page_icon="🎬", layout="centered")

st.markdown(
    """
    <div style="text-align: center;">
        <h1 style="color: #ff4b4b;">🎬 IMDB Movie Review Sentiment Analyzer</h1>
        <p style="font-size: 18px;">Paste a movie review below and see whether it's <b>Positive</b> or <b>Negative</b> 🎭</p>
    </div>
    """,
    unsafe_allow_html=True
)

# User input
user_input = st.text_area(
    'Enter your movie review here:',
    placeholder="E.g., This movie was fantastic! The acting was brilliant."
)

if st.button("🔍 Analyze Sentiment"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a movie review before analyzing.")
    else:
        preprocess_input = preprocess_text(user_input)
        prediction = model.predict(preprocess_input)
        sentiment = "Positive" if prediction[0][0] > 0.5 else "Negative"
        score = prediction[0][0]

        if sentiment == "Positive":
            st.success(f"✅ **Sentiment:** {sentiment} ({score:.2f})")
        else:
            st.error(f"❌ **Sentiment:** {sentiment} ({score:.2f})")

        with st.expander("🔍 See decoded review tokens"):
            decoded = decode_review(preprocess_input[0])
            st.write(decoded)

else:
    st.info("💡 Enter a review and click **Analyze Sentiment** to begin.")
