# Student Placement Classifier

A small Flask application that predicts whether a student is likely to be placed using a scikit-learn decision tree.

## Setup

From the `placement-classifier` directory:

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
python model/train_model.py
python backend/app.py
```

Open `http://127.0.0.1:5000` in a browser. The training script creates `model/model.pkl`, `model/feature_importance.json`, and `model/metrics.json`.

The `/predict` endpoint accepts JSON or form data with `cgpa`, `internships`, `projects`, `communication_score`, and `aptitude_score`.