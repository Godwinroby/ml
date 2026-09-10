import os
import pandas as pd

# Set the path to your datasets folder
folder_path = r"C:\Users\Godwin Arulraj\Desktop\ml\datasets"

# Get all CSV files in the folder
csv_files = [f for f in os.listdir(folder_path) if f.endswith(".csv")]

dataframes = []

print("🔍 Found files:\n", "\n".join(csv_files))

for file in csv_files:
    path = os.path.join(folder_path, file)
    try:
        try:
            df = pd.read_csv(path, encoding='utf-8', on_bad_lines='skip')
        except:
            df = pd.read_csv(path, encoding='latin1', on_bad_lines='skip')
        
        df.columns = df.columns.str.lower().str.strip()

        # --- Handle text_combined case (e.g., phishing_email.csv) ---
        if "text_combined" in df.columns:
            df["body"] = df["text_combined"]
            df["subject"] = df["text_combined"].str.extract(r"Subject:\s*(.*)", expand=False)
            df["subject"] = df["subject"].fillna("")  # fallback if no match

        # Fill missing subject/body with blank
        if "subject" not in df.columns:
            df["subject"] = ""
        if "body" not in df.columns:
            df["body"] = df["email"] if "email" in df.columns else ""
        if "body" not in df.columns:
            df["body"] = ""

        # Normalize label column
        label_col = None
        for col in ["label", "spam", "target", "class"]:
            if col in df.columns:
                label_col = col
                break

        if label_col:
            df["label"] = df[label_col]
        else:
            df["label"] = 0  # default if missing

        # Final label normalization (spam, phishing, etc → 1; ham/clean → 0)
        df["label"] = df["label"].apply(lambda x: 1 if str(x).strip().lower() in ["1", "phishing", "spam", "fraud", "scam"] else 0)

        # Select and reorder columns
        df_final = df[["subject", "body", "label"]].copy()
        df_final["source"] = file
        df_final.dropna(subset=["body"], inplace=True)
        df_final = df_final[df_final["body"].str.len() > 20]

        dataframes.append(df_final)
        print(f"✅ Processed {file}: {len(df_final)} rows")

    except Exception as e:
        print(f"❌ Failed to process {file}: {e}")

# 🔁 Combine all
if dataframes:
    combined_df = pd.concat(dataframes, ignore_index=True)
    combined_df.drop_duplicates(subset="body", inplace=True)
    combined_df = combined_df.sample(frac=1, random_state=42).reset_index(drop=True)

    output_file = os.path.join(folder_path, "combinedemailsdataset.csv")
    combined_df.to_csv(output_file, index=False)

    print("\n🎉 Combined dataset saved at:")
    print(output_file)
    print(f"📊 Rows: {len(combined_df)} | 🧪 Phishing: {combined_df['label'].sum()} | ✅ Ham: {(combined_df['label'] == 0).sum()}")
else:
    print("❌ No valid CSVs were combined. Please check your data.")
