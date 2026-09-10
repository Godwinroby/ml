import pandas as pd
import os

# 📁 Set your dataset folder path here
data_dir = r"C:\Users\Godwin Arulraj\Desktop\ml\datasets"

# ✅ List only the CSVs that actually exist in your folder
csv_files = [
    "phishing_email.csv",
    "Enron.csv",
    "SpamAssassin.csv",
    "CEAS_08.csv",
    "Ling.csv",
    "Nazario.csv",
    "Nigerian_Fraud.csv"
]

combined_data = []

print("🔍 Looking for the following files:")
for file in csv_files:
    print(" -", os.path.join(data_dir, file))

# 🔄 Load and clean each CSV
for file_name in csv_files:
    file_path = os.path.join(data_dir, file_name)
    try:
        df = pd.read_csv(file_path, encoding='utf-8', on_bad_lines='skip')
        df = df.rename(columns=lambda x: x.strip())

        # Standardize column names
        if "Email Text" in df.columns:
            df = df.rename(columns={"Email Text": "text"})
        elif "email" in df.columns:
            df = df.rename(columns={"email": "text"})
        elif "Email" in df.columns:
            df = df.rename(columns={"Email": "text"})

        if "Label" not in df.columns and "label" in df.columns:
            df = df.rename(columns={"label": "Label"})

        # Only keep valid columns
        if "text" in df.columns and "Label" in df.columns:
            df = df[["text", "Label"]]
            df.dropna(inplace=True)
            df = df[df["text"].str.len() > 20]  # Remove short/junk
            df["Label"] = df["Label"].apply(lambda x: 1 if str(x).strip().lower() in ["1", "spam", "phishing", "fraud"] else 0)
            combined_data.append(df)
            print(f"✅ Loaded: {file_name} ({len(df)} rows)")
        else:
            print(f"⚠️ Skipped: {file_name} — missing required columns")

    except Exception as e:
        print(f"❌ Error loading {file_name} → {e}")

# 🧩 Combine everything
if combined_data:
    final_df = pd.concat(combined_data, ignore_index=True).drop_duplicates(subset="text").reset_index(drop=True)

    # 💾 Save final output
    output_path = os.path.join(data_dir, "final_combined_dataset.csv")
    final_df.to_csv(output_path, index=False)

    print("\n🎉 Combined dataset created successfully!")
    print(f"📦 File saved as: {output_path}")
    print(f"🔢 Total emails: {len(final_df)} | 🧪 Phishing: {final_df['Label'].sum()} | ✅ Ham: {(final_df['Label'] == 0).sum()}")
else:
    print("\n❌ No datasets were successfully loaded. Please check filenames and formatting.")
