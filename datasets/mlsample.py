import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score
from sklearn.pipeline import make_pipeline
import joblib

# ✅ Load dataset
df = pd.read_csv("combinedemailsdataset.csv")

# ✅ Combine subject + body
df['text'] = df['subject'].fillna('') + " " + df['body'].fillna('')
df['text'] = df['text'].str.lower()

# ✅ Use only 10,000 rows for faster training
df_small = df.sample(n=10000, random_state=42)
X = df_small['text']
y = df_small['label']

# ✅ TF-IDF
vectorizer = TfidfVectorizer(max_features=5000)
X_tfidf = vectorizer.fit_transform(X)

# ✅ Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X_tfidf, y, test_size=0.2, random_state=42)

# ✅ Base Learners
base_learners = [
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='logloss', verbosity=0)),
    ('cat', CatBoostClassifier(verbose=0)),
    ('lr', LogisticRegression(max_iter=1000))
]

# ✅ Final Estimator
final_estimator = RandomForestClassifier(n_estimators=100)

# ✅ Stacking Model
model = StackingClassifier(estimators=base_learners, final_estimator=final_estimator, cv=5)

print("📈 Training stacking model (XGB + CatBoost + LR → RF)...")
model.fit(X_train, y_train)

# ✅ Evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"✅ Accuracy on test set: {acc * 100:.2f}%")

# ✅ Save model
joblib.dump((model, vectorizer), "trained_model.pkl")
print("💾 Model saved as trained_model.pkl")
