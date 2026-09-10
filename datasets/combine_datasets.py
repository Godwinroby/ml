import os
import pandas as pd

# Your folder path where CSV files are extracted
folder_path = r"C:\Users\Godwin Arulraj\Desktop\ml\datasets"

# Get all CSV files in the folder
csv_files = [file for file in os.listdir(folder_path) if file.endswith(".csv")]

# List to hold DataFrames
dataframes = []

for file in csv_files:
    file_path = os.path.join(folder_path, file)
    try:
        df = pd.read_csv(file_path, encoding='utf-8')
    except:
        df = pd.read_csv(file_path, encoding='latin1')  # fallback encoding

    # Normalize to subject, body, label
    if 'text_combined' in df.columns:
        df['subject'] = df['text_combined'].str.extract(r'^(.*?)(?=\s)', expand=False)
        df['body'] = df['text_combined']
    if 'subject' not in df.columns:
        df['subject'] = ''
    if 'body' not in df.columns:
        df['body'] = ''
    if 'label' not in df.columns:
        df['label'] = 0

    df = df[['subject', 'body', 'label']]
    df['source'] = file  # Optional: Track file origin
    dataframes.append(df)

# Combine all dataframes
combined_df = pd.concat(dataframes, ignore_index=True)

# Save to combined CSV
output_file = os.path.join(folder_path, "combined_emails_dataset.csv")
combined_df.to_csv(output_file, index=False)

print(f"✅ Combined CSV saved at: {output_file}")
