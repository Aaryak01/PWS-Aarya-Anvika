import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_excel("merged_maternal_clean.xlsx")

# 80/20 split, stratified on risklevel to keep class balance
train, test = train_test_split(df, test_size=0.2, random_state=42, stratify=df['risklevel'])

# Export
train.to_excel("maternal_health_train.xlsx", index=False)
test.to_excel("maternal_health_test.xlsx", index=False)

print(f"Train rows: {len(train)}")
print(f"Test rows:  {len(test)}")
print("\nRisk level distribution in train:")
print(train['risklevel'].value_counts())
print("\nRisk level distribution in test:")
print(test['risklevel'].value_counts())