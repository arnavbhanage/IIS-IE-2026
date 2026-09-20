"""Train and save a decision-tree student placement classifier."""

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "student_placement_data.csv"
MODEL_PATH = ROOT / "model" / "model.pkl"
IMPORTANCE_PATH = ROOT / "model" / "feature_importance.json"
METRICS_PATH = ROOT / "model" / "metrics.json"
FEATURES = [
    "cgpa",
    "internships",
    "projects",
    "communication_score",
    "aptitude_score",
]


def train() -> None:
    data = pd.read_csv(DATA_PATH)
    features = data[FEATURES]
    target = data["placed"]
    x_train, x_test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=42, stratify=target
    )

    classifier = DecisionTreeClassifier(max_depth=4, random_state=42)
    classifier.fit(x_train, y_train)
    predictions = classifier.predict(x_test)

    joblib.dump(classifier, MODEL_PATH)
    importance = dict(zip(FEATURES, classifier.feature_importances_))
    IMPORTANCE_PATH.write_text(json.dumps(importance, indent=2), encoding="utf-8")
    metrics = {
        "accuracy": accuracy_score(y_test, predictions),
        "classification_report": classification_report(
            y_test, predictions, output_dict=True, zero_division=0
        ),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Saved model to {MODEL_PATH}")
    print(f"Test accuracy: {metrics['accuracy']:.2f}")


if __name__ == "__main__":
    train()