import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import (
    StandardScaler,
    OneHotEncoder,
    LabelEncoder
)
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

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
# ENCODE TARGET
# ==================================================

label_encoder = LabelEncoder()

y_train_encoded = label_encoder.fit_transform(y_train)
y_test_encoded = label_encoder.transform(y_test)

print("Classes:")
print(label_encoder.classes_)

# ==================================================
# SAME FEATURES AS LOGISTIC REGRESSION
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
# PREPROCESS DATA
# FIT ONLY ON TRAINING DATA
# ==================================================

X_train_processed = preprocessor.fit_transform(X_train)
X_test_processed = preprocessor.transform(X_test)

# ==================================================
# NEURAL NETWORK
# ==================================================

model = MLPClassifier(
    hidden_layer_sizes=(256, 128, 64),
    activation="relu",
    solver="adam",
    learning_rate_init=0.001,
    max_iter=500,
    random_state=42,
    verbose=True
)

# ==================================================
# TRAIN
# ==================================================

print("\nTraining Neural Network...")

model.fit(X_train_processed, y_train_encoded)

print("\nTraining Complete!")

# ==================================================
# PREDICT
# ==================================================

y_pred = model.predict(X_test_processed)

# ==================================================
# EVALUATE
# ==================================================

accuracy = accuracy_score(y_test_encoded, y_pred)

print("\n==============================")
print("NEURAL NETWORK PERFORMANCE")
print("==============================")

print(f"\nAccuracy: {accuracy:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_encoded, y_pred))

print("\nClassification Report:")
print(
    classification_report(
        y_test_encoded,
        y_pred,
        target_names=label_encoder.classes_
    )
)

# ==================================================
# TRAINING CURVE
# ==================================================

plt.figure(figsize=(10, 5))
plt.plot(model.loss_curve_)

plt.title("Neural Network Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")

plt.savefig("nn_training_curve.png")
plt.show()

print("\nTraining curve saved as nn_training_curve.png")