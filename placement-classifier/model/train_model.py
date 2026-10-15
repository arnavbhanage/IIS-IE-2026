from pathlib import Path
import json
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import joblib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
FEATURE_COLUMNS = ['cgpa', 'attendance_percent', 'leadership_roles', 'academic_gaps', 'project_complexity', 'internships', 'coding_questions_solved']

def demo_category(row):
    # Explicit assumptions, not observed placement outcomes. Extra roles and
    # internships do not earn additional points beyond the first.
    practical = (2 * (row['internships'] >= 1) + (row['coding_questions_solved'] >= 150)
                 + (row['project_complexity'] >= 3) + (row['leadership_roles'] >= 1))
    if row['cgpa'] >= 8.5 and practical >= 2:
        return '1'
    if row['cgpa'] >= 7 and practical >= 1:
        return '2'
    return '3'

def main():
    rng = np.random.default_rng(42)
    n = 6000
    df = pd.DataFrame({
        'cgpa': rng.integers(40, 101, n) / 10,
        'attendance_percent': rng.integers(50, 101, n),
        'leadership_roles': rng.choice([0, 1, 2, 3], n, p=[.79, .18, .025, .005]),
        'academic_gaps': rng.choice([0, 1, 2, 3], n, p=[.88, .09, .025, .005]),
        'project_complexity': rng.choice([1, 2, 3, 4, 5], n, p=[.15, .25, .35, .2, .05]),
        'internships': rng.choice([0, 1, 2], n, p=[.65, .33, .02]),
        'coding_questions_solved': np.minimum(rng.gamma(1.5, 95, n).astype(int), 1000),
    })
    df['placement_category'] = df.apply(demo_category, axis=1)
    df = df.drop_duplicates()
    df.to_csv(ROOT / 'data/college_demo_data.csv', index=False)
    train, test = train_test_split(df, test_size=.2, random_state=42, stratify=df.placement_category)
    model = DecisionTreeClassifier(max_depth=6, min_samples_leaf=8, random_state=42)
    model.fit(train[FEATURE_COLUMNS], train.placement_category)
    predicted = model.predict(test[FEATURE_COLUMNS])
    joblib.dump(model, ROOT / 'model/model.pkl')
    importance = dict(zip(FEATURE_COLUMNS, model.feature_importances_.tolist()))
    (ROOT / 'model/feature_importance.json').write_text(json.dumps(importance, indent=2))
    metrics = {'accuracy':float(accuracy_score(test.placement_category, predicted)),
               'train_rows':len(train), 'test_rows':len(test), 'classes':model.classes_.tolist(),
               'confusion_matrix':confusion_matrix(test.placement_category, predicted, labels=model.classes_).tolist(),
               'max_depth':model.get_depth(), 'data_source':'Synthetic college demo rubric; not real placement outcomes.'}
    (ROOT / 'model/metrics.json').write_text(json.dumps(metrics, indent=2))
    plt.figure(figsize=(28, 14))
    plot_tree(model, feature_names=FEATURE_COLUMNS, class_names=model.classes_, filled=True, rounded=True, fontsize=6)
    plt.savefig(ROOT / 'model/tree_diagram.png', dpi=120, bbox_inches='tight')
    plt.close()
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
