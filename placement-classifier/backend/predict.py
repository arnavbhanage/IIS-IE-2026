"""
predict.py
Loads the trained Decision Tree and produces both a prediction AND a
human-readable explanation of *why*, by walking the tree's own decision path.
This is the core "explainable AI" piece — no SHAP/LIME/API needed.
"""

import os
import json
import joblib
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model.pkl")
FEATURES_PATH = os.path.join(BASE_DIR, "model", "feature_importance.json")

FEATURE_COLUMNS = [
    "cgpa",
    "attendance_percent",
    "leadership_roles",
    "academic_gaps",
    "project_complexity",
    "internships",
    "coding_questions_solved",
]

FEATURE_LABELS = {
    "cgpa": "CGPA",
    "attendance_percent": "Attendance %",
    "leadership_roles": "Leadership roles",
    "academic_gaps": "Academic gaps",
    "project_complexity": "Project complexity",
    "internships": "Internships",
    "coding_questions_solved": "Coding questions solved",
}

_model = joblib.load(MODEL_PATH)
with open(FEATURES_PATH) as f:
    _global_importances = json.load(f)


def _build_input_vector(data: dict):
    """Turns the incoming JSON dict into an ordered feature vector."""
    return np.array([[float(data[col]) for col in FEATURE_COLUMNS]])


def _explain_decision_path(x):
    """
    Walks the exact sequence of splits the tree used for this one input,
    producing a plain-English trail like:
    'CGPA (8.5) > 7.75 -> Internships (2) > 1 -> predicted A'
    """
    tree = _model.tree_
    node_indicator = _model.decision_path(x)
    leaf_id = _model.apply(x)[0]
    node_index = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]

    steps = []
    for node_id in node_index:
        if leaf_id == node_id:
            continue  # leaf node has no split, skip
        feature_idx = tree.feature[node_id]
        threshold = tree.threshold[node_id]
        feature_name = FEATURE_COLUMNS[feature_idx]
        value = x[0, feature_idx]

        if value <= threshold:
            direction = f"{FEATURE_LABELS[feature_name]} ({value:g}) <= {threshold:.2f}"
        else:
            direction = f"{FEATURE_LABELS[feature_name]} ({value:g}) > {threshold:.2f}"
        steps.append(direction)

    return steps


def _local_feature_contributions(x):
    """
    Simple local explanation: for each feature actually used along this
    student's decision path, report how many times / how strongly it
    contributed (weighted by the split's impurity decrease at that node).
    This gives a per-student bar-chart-friendly breakdown.
    """
    tree = _model.tree_
    node_indicator = _model.decision_path(x)
    node_index = node_indicator.indices[
        node_indicator.indptr[0] : node_indicator.indptr[1]
    ]

    contributions = {col: 0.0 for col in FEATURE_COLUMNS}
    for node_id in node_index:
        feature_idx = tree.feature[node_id]
        if feature_idx < 0:
            continue  # leaf
        feature_name = FEATURE_COLUMNS[feature_idx]
        # impurity decrease at this node = weighted_n_node_samples * impurity
        # minus children's weighted impurity. Use a simple proxy: impurity at
        # this node scaled by samples, which reflects how "decisive" the
        # split was overall (global, not sample-specific, but a reasonable
        # simple proxy for a basic project).
        contributions[feature_name] += tree.impurity[node_id] * tree.n_node_samples[node_id]

    total = sum(contributions.values()) or 1.0
    normalized = {k: round(v / total, 3) for k, v in contributions.items() if v > 0}
    return dict(sorted(normalized.items(), key=lambda kv: kv[1], reverse=True))


def predict_with_explanation(data: dict):
    x = _build_input_vector(data)

    prediction = _model.predict(x)[0]
    probabilities = dict(zip(_model.classes_, _model.predict_proba(x)[0].round(3).tolist()))

    decision_path = _explain_decision_path(x)
    local_contributions = _local_feature_contributions(x)

    return {
        "prediction": prediction,
        "probabilities": probabilities,
        "decision_path": decision_path,
        "local_feature_contributions": local_contributions,
        "global_feature_importance": _global_importances,
    }
