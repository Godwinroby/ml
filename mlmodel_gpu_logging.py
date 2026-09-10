import os
import re
import joblib
import logging
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
import seaborn as sns

# 📝 Setup logging  
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = f"training_log_{timestamp}.txt"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)
logging.info("🚀 Training script started")

# 📂 Load dataset
file_path = r"C:\Users\Godwin Arulraj\Desktop\ml\datasets\combinedemailsdataset.csv"
df = pd.read_csv(file_path)
logging.info(f"✅ Dataset loaded with shape: {df.shape}")

# 🧼 Preprocess text
df.fillna("", inplace=True)
df["text"] = df["subject"] + " " + df["body"]

def clean(text):
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = re.sub(r"[^a-zA-Z ]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.lower()

df["text"] = df["text"].apply(clean)
logging.info("✅ Text cleaned")

# ✨ TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(df["text"])
y = df["label"]
logging.info("✅ TF-IDF vectorization completed")

# 🎲 Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
logging.info(f"✅ Split complete: Train = {X_train.shape}, Test = {X_test.shape}")

# ⚡️ Define base models (with GPU)
xgb = XGBClassifier(tree_method='gpu_hist', predictor='gpu_predictor', gpu_id=0,
                    use_label_encoder=False, eval_metric='logloss', verbosity=0)
cat = CatBoostClassifier(task_type='GPU', devices='0', verbose=0)
lr = LogisticRegression(max_iter=1000)

# 🌲 Final estimator
rf = RandomForestClassifier(n_estimators=150, random_state=42)

# 🧠 Stacking model
stacked_model = StackingClassifier(
    estimators=[
        ('xgb', xgb),
        ('cat', cat),
        ('lr', lr)
    ],
    final_estimator=rf,
    passthrough=False
)

# 📈 Train
logging.info("📈 Training stacking model (XGB + CatBoost + LR → RF)...")
stacked_model.fit(X_train, y_train)
logging.info("✅ Model training completed")

# 🧪 Evaluate
y_pred = stacked_model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, digits=4)

logging.info(f"✅ Accuracy: {acc:.4f}")
logging.info("\n" + report)

# 📊 Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Ham', 'Phishing'], yticklabels=['Ham', 'Phishing'])
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
logging.info("🖼️ Confusion matrix saved as confusion_matrix.png")

# 💾 Save model + vectorizer
model_path = r"C:\Users\Godwin Arulraj\Desktop\ml\trained_model.pkl"
joblib.dump((stacked_model, vectorizer), model_path)
logging.info(f"💾 Model saved to: {model_path}")

