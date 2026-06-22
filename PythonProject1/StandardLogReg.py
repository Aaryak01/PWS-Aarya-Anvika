import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load data
train = pd.read_excel("maternal_health_train.xlsx")
test = pd.read_excel("maternal_health_test.xlsx")

# Clean labels
train["risklevel"] = train["risklevel"].replace({
    "low": "low risk",
    "high": "high risk"
})

test["risklevel"] = test["risklevel"].replace({
    "low": "low risk",
    "high": "high risk"
})

# Features and target
y_train = train["risklevel"]
X_train = train.drop(columns=["risklevel"])

y_test = test["risklevel"]
X_test = test.drop(columns=["risklevel"])

# Feature groups
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

# Preprocessing
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

# Standard Logistic Regression
model = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(
        max_iter=5000,
        random_state=42,
        solver="lbfgs",
    ))
])

# Train
model.fit(X_train, y_train)

# Predict
y_pred = model.predict(X_test)

# Evaluate
print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))