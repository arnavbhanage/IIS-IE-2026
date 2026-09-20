# Student Placement Category Predictor

A basic, explainable ML project that classifies students into placement
readiness categories (A / B / C) using a Decision Tree, with a Flask
backend and a plain HTML/CSS/JS frontend.

## Why a Decision Tree?

Explainability was a core requirement, so instead of using a "black box"
model + a separate explainability library (SHAP/LIME) or an external API,
a single shallow Decision Tree was chosen because **the model's own
structure is the explanation**. Every prediction can be traced as a
literal sequence of if/else splits (see `backend/predict.py` ->
`_explain_decision_path`), with zero external dependencies or API calls.

## Features used

- CGPA (0-10)
- Attendance %
- Leadership roles (count)
- Academic gaps (years/backlogs)
- Project complexity (1-5)
- Internships (count)
- Coding questions solved (LeetCode/HackerRank)

## Project structure

```
placement-classifier/
├── data/
│   └── student_placement_data.csv   # 540 synthetic student records
├── model/
│   ├── train_model.py                # cleans data, trains tree, saves model
│   ├── model.pkl                     # trained model (generated)
│   ├── feature_importance.json       # global feature importances (generated)
│   └── tree_diagram.png              # visual of the tree, for slides (generated)
├── backend/
│   ├── app.py                        # Flask routes
│   └── predict.py                    # prediction + explanation logic
├── frontend/
│   ├── templates/index.html
│   └── static/{style.css, script.js}
└── README.md
```

## Setup

```bash
pip install flask scikit-learn pandas joblib matplotlib

# 1. Train the model (run once, or whenever data/CSV changes)
cd model
python3 train_model.py

# 2. Run the Flask app
cd ../backend
python3 app.py
```

Then open http://127.0.0.1:5000 in your browser.

## How it works

1. `train_model.py` loads the CSV, does basic cleaning (dedupe, drop nulls,
   enforce logical ranges like CGPA 0-10), trains a `DecisionTreeClassifier`
   (`max_depth=4` to stay interpretable and avoid overfitting), and saves
   the model + feature importances + a tree diagram image.
2. `app.py` serves the form and exposes `POST /predict`.
3. `predict.py` loads the saved model and, for each prediction, also
   returns:
   - **Decision path** — the exact splits this student's input triggered
   - **Local feature contribution** — which features were most decisive
     for *this* student's path
   - **Global feature importance** — which features matter most across
     the whole model
   - **Class probabilities** — the tree's confidence in A/B/C
4. The frontend renders the category plus all of the above as readable
   bars and a step-by-step trail.

## Notes for the presentation

- The dataset is synthetic, generated from a weighted scoring formula with
  added random noise (so classes aren't perfectly separable — more
  realistic than a rule the tree could memorize exactly).
- Test accuracy is ~57%, with most confusion in the B class, which is
  expected since B sits between two threshold boundaries and is
  inherently the hardest to separate — worth mentioning as an honest
  limitation rather than hiding it.
- Attendance and leadership roles turned out to have the lowest global
  importance in this dataset — a good discussion point ("does attendance
  really predict placement, or did our data-generation formula underweight
  it deliberately?").
