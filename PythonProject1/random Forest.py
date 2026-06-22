import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# ==================================================
# LOAD DATA
# ==================================================

train = pd.read_excel("maternal_health_train.xlsx")
test = pd.read_excel("maternal_health_test.xlsx")

# ==================================================
# CLEAN RISK LABELS
# ==================================================

train["risklevel"] = train["risklevel"].replace({
    "low": "low risk",
    "high": "high risk"
})

test["risklevel"] = test["risklevel"].replace({
    "low": "low risk",
    "high": "high risk"
})

# ==================================================
# TARGET AND FEATURES
# ==================================================

y_train = train["risklevel"]
X_train = train.drop(columns=["risklevel"])

y_test = test["risklevel"]
X_test = test.drop(columns=["risklevel"])

# ==================================================
# SAME FEATURES AS LOGISTIC REGRESSION & NEURAL NETWORK
# ==================================================

numeric_features = [
    "age",
    "systolicbp",
    "diastolicbp",
    "bs",
    "bodytemp",
    "heartrate",
    "bmi"
]

categorical_features = [
    "previous_complications",
    "preexisting_diabetes",
    "gestational_diabetes",
    "mental_health"
]

# ==================================================
# PREPROCESSING
# ==================================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer, numeric_features),
    ("cat", categorical_transformer, categorical_features)
])

# ==================================================
# RANDOM FOREST
# ==================================================

model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced"
    ))
])

# ==================================================
# TRAIN
# ==================================================

print("Training Random Forest...")
model.fit(X_train, y_train)
print("Training Complete!")

# ==================================================
# PREDICT
# ==================================================

y_pred = model.predict(X_test)

# ==================================================
# EVALUATE
# ==================================================

print("\n==============================")
print("RANDOM FOREST PERFORMANCE")
print("==============================")

print(f"\nAccuracy: {accuracy_score(y_test, y_pred):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# ==================================================
# FEATURE IMPORTANCE PLOT
# ==================================================

rf = model.named_steps["classifier"]
ohe_features = list(
    model.named_steps["preprocessor"]
    .named_transformers_["cat"]
    .named_steps["encoder"]
    .get_feature_names_out(categorical_features)
)
all_features = numeric_features + ohe_features

importances = pd.Series(rf.feature_importances_, index=all_features).sort_values(ascending=True)

plt.figure(figsize=(10, 6))
importances.plot(kind="barh")
plt.title("Random Forest - Feature Importances")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig("rf_feature_importance.png")
plt.show()

print("\nFeature importance plot saved as rf_feature_importance.png")