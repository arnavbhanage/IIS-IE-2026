# Placement Lab

A local college project: enter a student profile, predict Category 1 / 2 / 3,
then read a plain-language explanation or inspect the actual decision tree path.
Flask serves a built React + shadcn/ui interface. No login, landing page,
external AI service or deployment is needed.

## Run the demo (PowerShell, from the repository root)

The current workspace already has a Python virtual environment. For a fresh checkout:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r placement-classifier/requirements.txt
```

Build the frontend once:

```powershell
cd placement-classifier/frontend
npm ci
npm run build
cd ../..
```

Train and run:

```powershell
.venv/Scripts/python.exe placement-classifier/model/train_model.py
.venv/Scripts/python.exe placement-classifier/backend/app.py
```

Open http://127.0.0.1:5000. After building, only Flask needs to run;
there are no CDN dependencies at runtime. Node is only needed to build/edit the UI.
For frontend development, run `npm run dev` in `placement-classifier/frontend`
with Flask running in a second terminal. Vite proxies prediction requests to Flask.

## Does the project make sense?

Yes, as an explainable classification demonstration. A shallow decision tree
is appropriate because the exact sequence of splits can be shown directly.
However, this is a simulation of a college readiness rubric, not a validated
prediction of employment or an official college category system. A rule-based
calculator could implement the rubric exactly; the educational purpose of the
ML version is to demonstrate learning, held-out evaluation, and explanations.
To claim real predictive value, replace synthetic labels with actual historical
placement outcomes, define categories with the college, and evaluate on unseen cohorts.

## Explicit assumptions

Category 1 means strong readiness, Category 2 developing readiness, and
Category 3 foundation building. The old A/B/C labels have been replaced.

Synthetic labels use this deliberately simple rubric:

- At least one internship: 2 practical points.
- At least 150 unique coding problems: 1 point.
- Project complexity at least 3: 1 point.
- At least one leadership role: 1 point.
- Category 1: CGPA >= 8.5 and practical points >= 2.
- Category 2: otherwise, CGPA >= 7 and practical points >= 1.
- Category 3: all remaining profiles.

One internship therefore supports Category 1 with 8.5+ CGPA; it does not
place every profile in Category 1 regardless of academics. Multiple internships
and leadership roles are allowed but earn no extra points beyond the first.
Internship counts are sampled 65% zero, 33% one, 2% two. Leadership counts
are 79% zero, 18% one, 2.5% two, 0.5% three. These are demo assumptions based
on the user's college context, not measured statistics.

Attendance and gap years remain available inputs but do not determine the
synthetic labels. Academic gaps mean years, not backlogs. The original CSV
is preserved as `data/student_placement_data.csv`; the reproducible generator
writes `data/college_demo_data.csv` and trains on that new data.

Project scale: 1 tutorial, 2 basic app, 3 independently built project,
4 advanced project, 5 research-oriented project. This is a subjective input.

## Coding benchmark research

[LeetCode Top Interview 150](https://leetcode.com/studyplan/top-interview-150/)
provides 150 classic questions spanning interview topics and describes itself
as suited to 3+ months of preparation. It supports a practical study reference,
not a placement cutoff. The 150 threshold above is our demo choice. Question
count does not capture difficulty, independence, retention, or contest ability.
Contest rating is a different measure and should not be inferred from solved count.

## Explanations

Both views use the same trained tree:

- Simple explanation: category, checks written as sentences, and training
  profile count in the final leaf. Unused inputs are explicitly listed.
- Model details: exact path (thresholds displayed to four decimals), class
  proportions in the leaf, local path importance, and global importance.

Local importance now sums the actual weighted impurity reduction at visited
splits, then normalizes it. The previous calculation used only the parent
impurity and mislabeled it as a contribution. These bars are a path-importance
heuristic, not SHAP values, causal effects, or positive/negative contributions.
Class proportions are not calibrated placement chances.

An explanation API is intentionally unnecessary for the working demo. The text
is generated from the actual tree, is repeatable, and works offline. If added
later, an LLM should only rephrase these supplied facts, never invent reasons
or change the prediction, and should retain this explanation as a fallback.

## Evaluation and checks

Run `.venv/Scripts/python.exe -m unittest discover -s placement-classifier/tests -v`.
Tests cover representative profiles, one-internship Category 1 behavior,
additional experience, explanation structure, API metadata, and invalid JSON,
missing fields, out-of-range values, fractional counts, booleans and nonfinite numbers.
Run `npm run build` from the frontend directory to validate the UI build.

Training uses seed 42, a stratified 80/20 split, depth at most 6 and at least
8 training samples per leaf. Metrics are saved in `model/metrics.json` and
served at `/model-info`; the tree image is saved in `model/tree_diagram.png`.
The current held-out synthetic agreement is 99.75% (1,200 test rows).
This high score reflects learning a deterministic synthetic rubric and says
nothing about real-world placement accuracy. The model can differ from the
rubric near boundaries. There is no hidden rule override after prediction.

## Main files

- `model/train_model.py`: synthetic data generation, training and evaluation.
- `backend/predict.py`: validation, prediction and faithful explanations.
- `backend/app.py`: Flask UI, `/predict`, `/model-info`.
- `frontend/src/main.jsx`: direct input/result workflow.
- `frontend/src/components/ui/`: components installed from the shadcn registry.
- `frontend/templates/` and `frontend/static/`: retained original UI, no longer served.
