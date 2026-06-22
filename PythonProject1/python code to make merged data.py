import pandas as pd

# Load both datasets - note sep=';' for the Kaggle one
df1 = pd.read_csv("Maternal Health Risk Data Set.csv", sep=';')
df2 = pd.read_csv("datasetNotKaggle.csv")

# Standardize column names
df1.columns = df1.columns.str.strip().str.lower().str.replace(" ", "_")
df2.columns = df2.columns.str.strip().str.lower().str.replace(" ", "_")

# Stack them
merged = pd.concat([df1, df2], ignore_index=True)

# Clean
merged = merged.drop_duplicates()
merged['risklevel'] = merged['risklevel'].str.strip().str.lower()

numeric_cols = ['age', 'systolicbp', 'diastolicbp', 'bs', 'bodytemp', 'heartrate']
for col in numeric_cols:
    merged[col] = pd.to_numeric(merged[col], errors='coerce')

merged = merged.dropna()

# Export to Excel
merged.to_excel("merged_maternal_clean.xlsx", index=False)
print("Done! Total rows after cleaning:", len(merged))