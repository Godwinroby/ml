import streamlit as st
import joblib
import re
import traceback

st.set_page_config(page_title="Email Phishing Detector", page_icon="🛡️", layout="wide")
st.title("🛡️ Email Phishing Detector")
st.info("Loading models, this may take 10-30 seconds...")

detection_model_path = r"C:\Users\Godwin Arulraj\Desktop\ml\trained_model.pkl"
intent_model_path = r"C:\Users\Godwin Arulraj\Desktop\ml\intent_classifier.pkl"

try:
    with st.spinner('Loading ML models...'):
        detection_model, detection_vectorizer = joblib.load(detection_model_path)
        intent_model, intent_vectorizer = joblib.load(intent_model_path)
    st.success("Models loaded! Enter email details below.")
except Exception as e:
    st.error(f"Could not load models: {e}")
    st.code(traceback.format_exc())
    st.stop()

def clean(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

subject = st.text_input("Email Subject", placeholder="e.g. Urgent: Verify your account")
body = st.text_area("Email Body", height=150, placeholder="Paste the full email body here")

if st.button("Analyze Email (Phishing Detection)"):
    try:
        combined_text = f"{subject or ''} {body or ''}"
        st.write(f"Combined: {combined_text}")
        cleaned = clean(combined_text)
        st.write(f"Cleaned: {cleaned}")

        # Vectorization
        X_main = detection_vectorizer.transform([cleaned])
        st.write(f"Vectorized shape: {X_main.shape}")

        phishing_pred = detection_model.predict(X_main)[0]
        st.write(f"Phishing prediction: {phishing_pred}")

        if phishing_pred == 1:
            X_intent = intent_vectorizer.transform([cleaned])
            st.write(f"Intent vector shape: {X_intent.shape}")
            intent_pred = intent_model.predict(X_intent)[0]
            st.error(f"⚠️ PHISHING DETECTED! Intent: {intent_pred}")
        else:
            st.success("✅ This is NOT phishing (Ham email).")
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        st.code(traceback.format_exc())
        # Also print in terminal for deeper diagnosis
        print("Fatal error during prediction:")
        traceback.print_exc()
        st.stop()

st.caption("Email security tool powered by your ML models.")
import os
os.system("pause")
