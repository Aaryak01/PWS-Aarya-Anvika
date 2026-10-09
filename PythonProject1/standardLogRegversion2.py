import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, label_binarize
from sklearn.linear_model import LogisticRegression
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
# MODEL  (standard — no class_weight)
# SMOTE added to fix mid-risk imbalance
# ==================================================

model = ImbPipeline([
    ("preprocessor", preprocessor),
    ("smote",        SMOTE(random_state=42)),
    ("classifier",   LogisticRegression(
        max_iter=5000,
        random_state=42,
        solver="lbfgs",
    )),
])

# ==================================================
# CROSS-VALIDATION  (before final fit)
# ==================================================

print("Running 5-fold cross-validation...")
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy")

print(f"  CV accuracy per fold : {cv_scores.round(3)}")
print(f"  Mean CV accuracy     : {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")

# ==================================================
# TRAIN ON FULL TRAINING SET
# ==================================================

print("\nTraining Standard Logistic Regression (+ SMOTE)...")
model.fit(X_train, y_train)
print("Training Complete!")
joblib.dump(model, 'lr_standard.pkl')

# ==================================================
# PREDICT
# ==================================================

y_pred      = model.predict(X_test)
y_pred_prob = model.predict_proba(X_test)

# ==================================================
# EVALUATE
# ==================================================

print("\n==============================")
print("STANDARD LOGISTIC REGRESSION")
print("==============================")

print(f"\nAccuracy : {accuracy_score(y_test, y_pred):.4f}")

# ROC-AUC (one-vs-rest, macro average)
y_test_bin = label_binarize(y_test, classes=CLASSES)
roc_auc    = roc_auc_score(y_test_bin, y_pred_prob, multi_class="ovr", average="macro")
print(f"ROC-AUC  : {roc_auc:.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred, labels=CLASSES))

print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASSES))

# ==================================================
# ROC CURVES  (one per class)
# ==================================================

plt.figure(figsize=(8, 6))
for i, cls in enumerate(CLASSES):
    fpr, tpr, _ = roc_curve(y_test_bin[:, i], y_pred_prob[:, i])
    auc_cls     = roc_auc_score(y_test_bin[:, i], y_pred_prob[:, i])
    plt.plot(fpr, tpr, label=f"{cls} (AUC = {auc_cls:.2f})")

plt.plot([0, 1], [0, 1], "k--", linewidth=0.8)
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves – Standard Logistic Regression")
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("roc_standard_logreg.png", dpi=150)
plt.show()
print("ROC curve saved as roc_standard_logreg.png")

# ==================================================
# CV ACCURACY BAR CHART
# ==================================================

plt.figure(figsize=(7, 4))
plt.bar(range(1, 6), cv_scores, color="steelblue", alpha=0.8)
plt.axhline(cv_scores.mean(), color="red", linestyle="--", label=f"Mean = {cv_scores.mean():.3f}")
plt.xlabel("Fold")
plt.ylabel("Accuracy")
plt.title("Cross-Validation – Standard Logistic Regression")
plt.ylim(0, 1)
plt.legend()
plt.tight_layout()
plt.savefig("cv_standard_logreg.png", dpi=150)
plt.show()
print("CV chart saved as cv_standard_logreg.png")