"""Prediction and faithful explanations derived from the fitted tree."""
import json
import math
from pathlib import Path
import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = ['cgpa','attendance_percent','leadership_roles','academic_gaps','project_complexity','internships','coding_questions_solved']
FEATURE_LABELS = dict(zip(FEATURE_COLUMNS, ['CGPA','Attendance','Leadership roles','Academic gap years','Project complexity','Internships','Coding questions solved']))
LIMITS = {'cgpa':(0,10), 'attendance_percent':(0,100), 'leadership_roles':(0,10), 'academic_gaps':(0,10), 'project_complexity':(1,5), 'internships':(0,10), 'coding_questions_solved':(0,5000)}
_model = joblib.load(ROOT / 'model/model.pkl')
_global_importances = dict(zip(FEATURE_COLUMNS, _model.feature_importances_.tolist()))

def validate(data):
    if not isinstance(data, dict):
        raise ValueError('Send a JSON object containing the student inputs.')
    values = {}
    for key in FEATURE_COLUMNS:
        if key not in data:
            raise ValueError(f'{FEATURE_LABELS[key]} is required.')
        if isinstance(data[key], bool) or not isinstance(data[key], (int,float)):
            raise ValueError(f'{FEATURE_LABELS[key]} must be a number.')
        value = float(data[key])
        low, high = LIMITS[key]
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f'{FEATURE_LABELS[key]} must be between {low} and {high}.')
        if key not in ['cgpa','attendance_percent'] and not value.is_integer():
            raise ValueError(f'{FEATURE_LABELS[key]} must be a whole number.')
        values[key] = value
    return values

def predict_with_explanation(data):
    values = validate(data)
    x = pd.DataFrame([values], columns=FEATURE_COLUMNS)
    prediction = str(_model.predict(x)[0])
    probabilities = dict(zip(_model.classes_.tolist(), _model.predict_proba(x)[0].tolist()))
    tree = _model.tree_
    nodes = _model.decision_path(x).indices
    path, sentences = [], []
    weights = {key:0.0 for key in FEATURE_COLUMNS}
    for node in nodes:
        idx = tree.feature[node]
        if idx < 0:
            continue
        key = FEATURE_COLUMNS[idx]
        threshold = float(tree.threshold[node])
        value = values[key]
        # sklearn evaluates float32 features; use the same comparison.
        left = float(np.float32(value)) <= threshold
        path.append(f'{FEATURE_LABELS[key]} ({value:g}) {"<=" if left else ">"} {threshold:.4f}')
        if key not in ['cgpa','attendance_percent']:
            words = f'at most {math.floor(threshold)}' if left else f'at least {math.floor(threshold)+1}'
        else:
            words = f'{"at or below" if left else "above"} {threshold:.2f}'
        if key == 'internships':
            sentence = (f'You have completed {value:g} internship(s). The tree follows the group with at least one internship.'
                        if not left and threshold < 1 else
                        f'You have completed {value:g} internship(s). The tree follows the group with {words} internship(s).')
        elif key == 'cgpa':
            sentence = f'Your CGPA of {value:g} puts you in the group {words} CGPA in this tree.'
        elif key == 'coding_questions_solved':
            sentence = f'You have solved {value:g} coding problems, so the tree follows the group with {words} problems solved.'
        else:
            sentence = f'Your value for {FEATURE_LABELS[key].lower()} is {value:g}, so the tree follows the group with {words}.'
        sentences.append(sentence)
        l, r = tree.children_left[node], tree.children_right[node]
        gain = (tree.weighted_n_node_samples[node]*tree.impurity[node]
                - tree.weighted_n_node_samples[l]*tree.impurity[l]
                - tree.weighted_n_node_samples[r]*tree.impurity[r])
        weights[key] += max(0.0, float(gain))
    total = sum(weights.values()) or 1
    local = {k:v/total for k,v in sorted(weights.items(), key=lambda item:-item[1]) if v>0}
    unused = [FEATURE_LABELS[k] for k in FEATURE_COLUMNS if k not in local]
    leaf = int(_model.apply(x)[0])
    titles = {'1':'strong readiness', '2':'developing readiness', '3':'foundation building'}
    summary = f'The model places this profile in Category {prediction} ({titles[prediction]}). The deciding checks were ' + ', '.join(FEATURE_LABELS[k] for k in local) + '.'
    return {'prediction':prediction, 'probabilities':probabilities, 'decision_path':path,
            'local_feature_contributions':local, 'global_feature_importance':_global_importances,
            'explanation':{'summary':summary, 'steps':sentences,
                           'unused_features':unused, 'source':'Generated directly from the decision tree'},
            'leaf_samples':int(tree.n_node_samples[leaf]),
            'probability_note':'These are category proportions in the training leaf, not your chance of getting placed.',
            'local_importance_note':'Normalized impurity reduction at splits on this path; not a causal effect or a signed contribution.',
            'limitation':'This model learns a synthetic college rubric. It has not been validated on real student placements.'}
