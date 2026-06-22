import pandas as pd

# Load all datasets
df1 = pd.read_csv("Maternal Health Risk Data Set.csv", sep=';')
df2 = pd.read_csv("datasetNotKaggle.csv")
df3 = pd.read_csv("Dataset - Updated.csv")

# Standardize column names
df1.columns = df1.columns.str.strip().str.lower().str.replace(" ", "_")
df2.columns = df2.columns.str.strip().str.lower().str.replace(" ", "_")
df3.columns = df3.columns.str.strip().str.lower().str.replace(" ", "_")

# Rename df3 columns to match df1 and df2
df3 = df3.rename(columns={
    'systolic_bp': 'systolicbp',
    'diastolic': 'diastolicbp',
    'body_temp': 'bodytemp',
    'risk_level': 'risklevel',
    'heart_rate': 'heartrate'
})

# Stack all three
merged = pd.concat([df1, df2, df3], ignore_index=True)

# Clean
merged = merged.drop_duplicates()
merged['risklevel'] = merged['risklevel'].str.strip().str.lower()

numeric_cols = ['age', 'systolicbp', 'diastolicbp', 'bs', 'bodytemp', 'heartrate']
for col in numeric_cols:
    if col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')

merged = merged.dropna(subset=['age', 'systolicbp', 'risklevel'])

# Fill empty cells with NULL so it's clear in Excel
merged = merged.fillna("NULL")

# Export
merged.to_excel("merged_maternal_clean.xlsx", index=False)
print("Done! Total rows:", len(merged))
print("Columns:", merged.columns.tolist())