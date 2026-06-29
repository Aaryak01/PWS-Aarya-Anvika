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

# Remove duplicates
merged = merged.drop_duplicates()

# FIX: Standardize risk level labels (was the main issue)
merged['risklevel'] = merged['risklevel'].str.strip().str.lower()
merged['risklevel'] = merged['risklevel'].replace({
    'low risk': 'low',
    'high risk': 'high',
    'mid risk': 'mid'
})

# Convert numeric columns
numeric_cols = ['age', 'systolicbp', 'diastolicbp', 'bs', 'bodytemp', 'heartrate']
for col in numeric_cols:
    if col in merged.columns:
        merged[col] = pd.to_numeric(merged[col], errors='coerce')

# FIX: Remove obvious outliers
merged = merged[merged['age'].between(10, 60)]
merged = merged[merged['heartrate'] >= 30]
merged = merged[merged['diastolicbp'] <= 120]

# Remove biologically impossible BMI value
merged = merged[merged['bmi'] != 0.0]

# Drop rows missing the essentials
merged = merged.dropna(subset=['age', 'systolicbp', 'risklevel'])

# Keep NaN as actual NaN (not the string "NULL") — models handle NaN properly
# Export
merged.to_excel("merged_maternal_clean.xlsx", index=False)

print("Done! Total rows:", len(merged))
print("Columns:", merged.columns.tolist())
print("\nRisk level counts:")
print(merged['risklevel'].value_counts())
print("\nMissing values:")
print(merged.isnull().sum())