import joblib
import re

# === CLEANING FUNCTION ===
def clean(text):
    import re
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# === LOAD PHASE 1: PHISHING DETECTION MODEL ===
detection_model_path = r"C:\Users\Godwin Arulraj\Desktop\ml\trained_model.pkl"
detection_model, detection_vectorizer = joblib.load(detection_model_path)

# === LOAD PHASE 2: INTENT CLASSIFICATION MODEL ===
intent_model_path = r"C:\Users\Godwin Arulraj\Desktop\ml\intent_classifier.pkl"
intent_model, intent_vectorizer = joblib.load(intent_model_path)

# === MAIN FUNCTION ===
def predict_email(subject, body):
    combined_text = f"{subject or ''} {body or ''}"
    cleaned_text = clean(combined_text)

    # === PHASE 1: PHISHING DETECTION ===
    X_main = detection_vectorizer.transform([cleaned_text])
    phishing_pred = detection_model.predict(X_main)[0]

    if phishing_pred == 1:
        # === PHASE 2: PHISHING INTENT CLASSIFICATION ===
        X_intent = intent_vectorizer.transform([cleaned_text])
        intent_pred = intent_model.predict(X_intent)[0]
        return f"⚠️ Phishing Detected\n🧠 Intent: {intent_pred}"
    else:
        return "✅ Safe Email (Ham)"

# === ENTRY POINT ===
if __name__ == "__main__":
    subject = input("Enter email subject: ")
    body = input("Enter email body: ")
    result = predict_email(subject, body)
    print("\n🔎 Result:\n", result)
