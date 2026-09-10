import pandas as pd
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# === CLEANING FUNCTION ===
def clean(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower().strip()

# === LOAD AUTO-LABELED PHISHING INTENT DATA ===
df = pd.read_csv(r"C:\Users\Godwin Arulraj\Desktop\ml\datasets\phishing_intent_dataset.csv")
df.fillna("", inplace=True)

# Combine subject + body and clean
df["text"] = (df["subject"] + " " + df["body"]).apply(clean)

# === FEATURE EXTRACTION ===
vectorizer = TfidfVectorizer(max_features=3000)
X = vectorizer.fit_transform(df["text"])
y = df["intent"]

# === TRAIN-TEST SPLIT ===
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# === MODEL TRAINING ===
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)

# === EVALUATION ===
y_pred = clf.predict(X_test)
print("\n📊 Classification Report:\n")
print(classification_report(y_test, y_pred))

# === SAVE MODEL & VECTORIZER === 
output_path = r"C:\Users\Godwin Arulraj\Desktop\ml\intent_classifier.pkl"
joblib.dump((clf, vectorizer), output_path)
print(f"\n✅ Model saved to: {output_path}")


