
import streamlit as st
import joblib

# Load model and vectorizer
model = joblib.load("model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

st.set_page_config(page_title="AI Text Detector", layout="centered")
st.title("🧠 AI vs Human Text Detector")
st.write("Paste your text below to detect whether it's written by a human or an AI.")

text_input = st.text_area("Enter your text:")

if st.button("Analyze"):
    if text_input.strip() == "":
        st.warning("⚠️ Please enter some text.")
    else:
        text_vec = vectorizer.transform([text_input])
        prediction = model.predict(text_vec)[0]

        if prediction == 0:
            st.success("✅ Human-written text.")
        else:
            st.info("🤖 AI-generated text.")
