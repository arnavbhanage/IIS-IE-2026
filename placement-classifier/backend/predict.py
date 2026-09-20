"""Load the trained model and produce placement predictions."""

import json
from pathlib import Path

import joblib
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "model" / "model.pkl"
IMPORTANCE_PATH = ROOT / "model" / "feature_importance.json"
FEATURES = [
    "cgpa",
    "internships",
    "projects",
    "communication_score",
    "aptitude_score",
]


def predict_student(student_features: dict) -> dict:
    if not MODEL_PATH.exists():
        raise FileNotFoundError("model.pkl is missing. Run model/train_model.py first.")

    model = joblib.load(MODEL_PATH)
    values = {feature: float(student_features[feature]) for feature in FEATURES}
    frame = pd.DataFrame([values], columns=FEATURES)
    prediction = int(model.predict(frame)[0])
    probability = float(model.predict_proba(frame)[0][prediction])

    with IMPORTANCE_PATH.open(encoding="utf-8") as file:
        importances = json.load(file)
    strongest_feature = max(importances, key=importances.get)
    outcome = "likely to be placed" if prediction else "not currently likely to be placed"
    explanation = (
        f"The model predicts that this student is {outcome}. "
        f"The strongest learned factor in this model is {strongest_feature.replace('_', ' ')}."
    )
    return {
        "placed": bool(prediction),
        "label": "Placed" if prediction else "Not placed",
        "confidence": round(probability * 100, 2),
        "explanation": explanation,
    }