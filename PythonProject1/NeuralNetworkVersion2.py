import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder, label_binarize
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# ==================================================
# LOAD DATA
# ==================================================

train = pd.read_excel("maternal_health_train.xlsx")
test  = pd.read_excel("maternal_health_test.xlsx")

# ==================================================
# CLEAN RISK LABELS
# ==================================================

for df in [train, test]:
    df["risklevel"] = (
        df["risklevel"]
        .str.lower()
        .str.replace(" risk", "", regex=False)
        .str.strip()
    )

# ==================================================
# FEATURES & TARGET
# ==================================================

numeric_features = ["age", "systolicbp", "diastolicbp", "bs",
                    "bodytemp", "heartrate", "bmi"]

categorical_features = ["previous_complications", "preexisting_diabetes",
                        "gestational_diabetes", "mental_health"]

y_train = train["risklevel"]
X_train = train[numeric_features + categorical_features]

y_test = test["risklevel"]
X_test = test[numeric_features + categorical_features]

CLASSES = ["high", "low", "mid"]

# ==================================================
# LABEL ENCODE TARGET  (MLPClassifier works best with integers)
# ==================================================

le = LabelEncoder()
le.fit(CLASSES)
y_train_enc = le.transform(y_train)
y_test_enc  = le.transform(y_test)

print("Classes:", le.classes_)

# ==================================================
# PREPROCESSING
# ==================================================

numeric_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="mean")),
    ("scaler",  StandardScaler()),
])

categorical_transformer = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),
])

preprocessor = ColumnTransformer([
    ("num", numeric_transformer,     numeric_features),
    ("cat", categorical_transformer, categorical_features),
])

# ==================================================
# MODEL  (SMOTE to fix mid-risk imbalance)
# ==================================================

model = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote",        SMOTE(random_state=42)),
    ("classifier",   MLPClassifier(
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        learning_rate_init=0.001,
        max_iter=500,
        early_stopping=True,       # stops when validation loss stops improving
        validation_fraction=0.1,
        n_iter_no_change=10,
        random_state=42,
        verbose=False,
    )),
])

# ==================================================
# CROSS-VALIDATION
# ==================================================

print("Running 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train_enc, cv=cv, scoring="accuracy")

print(f"  CV accuracy per fold : {cv_scores.round(3)}")
print(f"  Mean CV accuracy     : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==================================================
# TRAIN ON FULL TRAINING SET
# ==================================================

print("\nTraining Neural Network (+ SMOTE)...")
model.fit(X_train, y_train_enc)
print("Training Complete!")

# ==================================================
# PREDICT
# ==================================================

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

# ==================================================
# EVALUATE
# ==================================================

print("\n==============================")
print("NEURAL NETWORK PERFORMANCE")
print("==============================")

print(f"\nAccuracy : {accuracy_score(y_test_enc, y_pred):.4f}")

y_test_bin = label_binarize(y_test_enc, classes=[0, 1, 2])
roc_auc    = roc_auc_score(y_test_bin, y_pred_prob, multi_class="ovr", average="macro")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test_enc, y_pred))

print("\nClassification Report:")
print(classification_report(y_test_enc, y_pred, target_names=le.classes_))

# ==================================================
# ROC CURVES
# ==================================================

plt.figure(figsize=(8, 6))
for i, cls in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    auc_cls     = roc_auc_score(y_test_bin[:, i], y_pred_prob[:, i])
    plt.plot(fpr, tpr, label=f"{cls} (AUC = {auc_cls:.2f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Neural Network")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_neural_network.png", dpi=150)
plt.show()
print("ROC curve saved as roc_neural_network.png")

# ==================================================
# TRAINING LOSS CURVE
# ==================================================

nn = model.named_steps["classifier"]
plt.figure(figsize=(10, 5))
plt.plot(nn.loss_curve_, label="Training loss")
if hasattr(nn, "validation_scores_"):
    plt.plot(nn.validation_scores_, label="Validation score", linestyle="--")
plt.title("Neural Network Training Loss")
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.savefig("nn_training_curve.png", dpi=150)
plt.show()
print("Training curve saved as nn_training_curve.png")

# ==================================================
# CV ACCURACY BAR CHART
# ==================================================

plt.figure(figsize=(7, 4))
plt.bar(range(1, 6), cv_scores, color="mediumpurple", alpha=0.8)
plt.axhline(cv_scores.mean(), color="red", linestyle="--", label=f"Mean = {cv_scores.mean():.3f}")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross-Validation – Neural Network")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("cv_neural_network.png", dpi=150)
plt.show()
print("CV chart saved as cv_neural_network.png")