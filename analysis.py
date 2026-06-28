"""
Real-World Data Project — Health Domain
Dataset: Breast Cancer Wisconsin (Diagnostic) Dataset
Goal: Predict whether a tumor is malignant or benign from patient diagnostic
measurements (an end-to-end classification task on real clinical data).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, confusion_matrix, roc_curve, roc_auc_score,
                              classification_report)

sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 140
OUT = "images/"

# ---------------------------------------------------------------
# 1. LOAD DATA
# ---------------------------------------------------------------
data = load_breast_cancer(as_frame=True)
df = data.frame.copy()
df["diagnosis"] = df["target"].map({0: "Malignant", 1: "Benign"})

print("Shape:", df.shape)
print(df["diagnosis"].value_counts())
df.to_csv("patient_records.csv", index=False)

# ---------------------------------------------------------------
# 2. EDA
# ---------------------------------------------------------------
summary = df.describe().T
summary.to_csv("summary_statistics.csv")

# Class balance plot
plt.figure(figsize=(5, 4))
ax = sns.countplot(data=df, x="diagnosis", palette=["#d65f5f", "#5fa8d6"])
ax.set_title("Diagnosis Class Distribution")
ax.set_xlabel("")
ax.set_ylabel("Number of Patients")
for p in ax.patches:
    ax.annotate(f"{int(p.get_height())}", (p.get_x() + p.get_width()/2, p.get_height()),
                ha="center", va="bottom")
plt.tight_layout()
plt.savefig(OUT + "class_distribution.png")
plt.close()

# Key feature distributions by diagnosis
key_features = ["mean radius", "mean texture", "mean perimeter", "mean area",
                 "mean smoothness", "mean concavity"]
fig, axes = plt.subplots(2, 3, figsize=(13, 7))
for ax, feat in zip(axes.flat, key_features):
    sns.kdeplot(data=df, x=feat, hue="diagnosis", fill=True, alpha=0.4, ax=ax,
                palette=["#d65f5f", "#5fa8d6"], legend=(feat == key_features[0]))
    ax.set_title(feat)
fig.suptitle("Distribution of Key Tumor Measurements by Diagnosis", y=1.02, fontsize=14)
plt.tight_layout()
plt.savefig(OUT + "feature_distributions.png", bbox_inches="tight")
plt.close()

# Correlation heatmap (mean features only, for readability)
mean_cols = [c for c in df.columns if c.startswith("mean")]
plt.figure(figsize=(9, 7))
corr = df[mean_cols].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, square=True,
            cbar_kws={"shrink": 0.8})
plt.title("Correlation Heatmap — Mean Tumor Features")
plt.tight_layout()
plt.savefig(OUT + "correlation_heatmap.png")
plt.close()

# Boxplot of a discriminating feature
plt.figure(figsize=(5, 4))
sns.boxplot(data=df, x="diagnosis", y="worst concave points", palette=["#d65f5f", "#5fa8d6"])
plt.title("Worst Concave Points by Diagnosis")
plt.tight_layout()
plt.savefig(OUT + "boxplot_worst_concave_points.png")
plt.close()

# ---------------------------------------------------------------
# 3. MODELING
# ---------------------------------------------------------------
X = df[data.feature_names]
y = df["target"]  # 0 = malignant, 1 = benign

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_s = scaler.fit_transform(X_train)
X_test_s = scaler.transform(X_test)

models = {
    "Logistic Regression": LogisticRegression(max_iter=5000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
}

results = []
roc_data = {}
for name, model in models.items():
    model.fit(X_train_s, y_train)
    preds = model.predict(X_test_s)
    proba = model.predict_proba(X_test_s)[:, 1]

    acc = accuracy_score(y_test, preds)
    prec = precision_score(y_test, preds)
    rec = recall_score(y_test, preds)
    f1 = f1_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    cv_scores = cross_val_score(model, X_train_s, y_train, cv=5)

    results.append({
        "Model": name, "Accuracy": acc, "Precision": prec,
        "Recall": rec, "F1-Score": f1, "ROC-AUC": auc,
        "CV Mean Accuracy": cv_scores.mean()
    })

    fpr, tpr, _ = roc_curve(y_test, proba)
    roc_data[name] = (fpr, tpr, auc)

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(4.5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["Malignant", "Benign"], yticklabels=["Malignant", "Benign"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    fname = name.lower().replace(" ", "_")
    plt.savefig(OUT + f"confusion_matrix_{fname}.png")
    plt.close()

    print(f"\n=== {name} ===")
    print(classification_report(y_test, preds, target_names=["Malignant", "Benign"]))

results_df = pd.DataFrame(results)
results_df.to_csv("model_results.csv", index=False)
print("\n", results_df)

# ROC curve comparison
plt.figure(figsize=(5.5, 5))
for name, (fpr, tpr, auc) in roc_data.items():
    plt.plot(fpr, tpr, label=f"{name} (AUC = {auc:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(OUT + "roc_curve.png")
plt.close()

# Feature importance (Random Forest)
rf = models["Random Forest"]
importances = pd.Series(rf.feature_importances_, index=data.feature_names).sort_values(ascending=False).head(10)
plt.figure(figsize=(7, 5))
sns.barplot(x=importances.values, y=importances.index, color="#5fa8d6")
plt.title("Top 10 Most Important Features (Random Forest)")
plt.xlabel("Importance")
plt.tight_layout()
plt.savefig(OUT + "feature_importance.png")
plt.close()

print("\nAll plots saved to images/. CSVs saved: patient_records.csv, summary_statistics.csv, model_results.csv")
