import pandas as pd

# ✅ Load your actual combined dataset
file_path = r"C:\Users\Godwin Arulraj\Desktop\ml\datasets\combined_emails_dataset.csv"
df = pd.read_csv(file_path)
df.fillna("", inplace=True)

# ✅ Only focus on phishing emails
df = df[df["label"] == 1].copy()

# 🔍 Check if 'source' column exists
if "source" not in df.columns:
    raise ValueError("❌ 'source' column not found in dataset. Add it when combining datasets.")

# ✅ Define source → intent mapping
def map_intent(source):
    source = source.lower()
    if "nigerian" in source:
        return "Financial Fraud"
    elif "nazario" in source:
        return "Malware Delivery"
    elif "phishing" in source:
        return "Credential Harvesting"
    elif "ceas" in source or "spamassassin" in source:
        return "Scam Offer"
    elif "enron" in source or "ling" in source:
        return None  # ham/irrelevant
    else:
        return None

# 🧠 Apply mapping
df["intent"] = df["source"].apply(map_intent)
df = df[df["intent"].notnull()].copy()

# 💾 Save the intent-labeled phishing dataset
output_path = r"C:\Users\Godwin Arulraj\Desktop\ml\datasets\phishing_intent_dataset.csv"
df[["subject", "body", "intent"]].to_csv(output_path, index=False)

print(f"✅ Auto-labeled phishing intent dataset saved at:\n{output_path}")
