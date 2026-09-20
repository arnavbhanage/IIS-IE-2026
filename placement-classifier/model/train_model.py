"""
train_model.py
Trains a Decision Tree classifier to predict student placement category (A/B/C).
Chosen for explainability: the tree's own splits ARE the explanation, no
external XAI library or API needed.
"""

import pandas as pd
import joblib
import json
import os

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
import matplotlib
matplotlib.use("Agg")  # no display needed, just save the figure
import matplotlib.pyplot as plt

# ---------- paths ----------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "student_placement_data.csv")
MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(MODEL_DIR, "model.pkl")
FEATURES_PATH = os.path.join(MODEL_DIR, "feature_importance.json")
TREE_IMG_PATH = os.path.join(MODEL_DIR, "tree_diagram.png")

FEATURE_COLUMNS = [
    "cgpa",
    "attendance_percent",
    "leadership_roles",
    "academic_gaps",
    "project_complexity",
    "internships",
    "coding_questions_solved",
]
TARGET_COLUMN = "placement_category"


def load_and_clean_data(path):
    df = pd.read_csv(path)

    # --- data cleaning ---
    # 1. drop exact duplicates
    df = df.drop_duplicates()

    # 2. drop rows with missing values (median-impute instead, if you want to
    #    keep every row — left simple here since the synthetic data has none)
    df = df.dropna()

    # 3. enforce logical ranges (defensive — guards against bad data entry)
    df = df[(df["cgpa"] >= 0) & (df["cgpa"] <= 10)]
    df = df[(df["attendance_percent"] >= 0) & (df["attendance_percent"] <= 100)]
    df = df[(df["project_complexity"] >= 1) & (df["project_complexity"] <= 5)]
    df = df[df["leadership_roles"] >= 0]
    df = df[df["academic_gaps"] >= 0]
    df = df[df["internships"] >= 0]
    df = df[df["coding_questions_solved"] >= 0]

    return df.reset_index(drop=True)


def main():
    df = load_and_clean_data(DATA_PATH)
    print(f"Loaded {len(df)} clean rows")
    print(df[TARGET_COLUMN].value_counts())

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    # stratified split so A/B/C proportions are preserved in train & test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Shallow tree on purpose: keeps it explainable and avoids overfitting
    # on a dataset this size.
    clf = DecisionTreeClassifier(
        max_depth=4,
        criterion="gini",
        random_state=42,
    )
    clf.fit(X_train, y_train)

    # ---------- evaluate ----------
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest accuracy: {acc:.3f}\n")
    print(classification_report(y_test, y_pred))
    print("Confusion matrix (rows=actual, cols=predicted), classes:", clf.classes_)
    print(confusion_matrix(y_test, y_pred, labels=clf.classes_))

    # ---------- save model ----------
    joblib.dump(clf, MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")

    # ---------- feature importance (global explainability) ----------
    importances = dict(zip(FEATURE_COLUMNS, clf.feature_importances_.tolist()))
    importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))
    with open(FEATURES_PATH, "w") as f:
        json.dump(importances, f, indent=2)
    print(f"Feature importances saved to {FEATURES_PATH}")
    for feat, imp in importances.items():
        print(f"  {feat:28s} {imp:.3f}")

    # ---------- save a visual of the tree for slides ----------
    plt.figure(figsize=(20, 10))
    plot_tree(
        clf,
        feature_names=FEATURE_COLUMNS,
        class_names=clf.classes_,
        filled=True,
        rounded=True,
        fontsize=9,
    )
    plt.savefig(TREE_IMG_PATH, dpi=150, bbox_inches="tight")
    print(f"Tree diagram saved to {TREE_IMG_PATH}")


if __name__ == "__main__":
    main()
