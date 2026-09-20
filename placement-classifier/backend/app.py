"""
app.py
Flask backend: serves the frontend form and a /predict endpoint that
returns a category (A/B/C) plus an explanation of the decision.
"""

import os
from flask import Flask, request, jsonify, render_template

from predict import predict_with_explanation, FEATURE_COLUMNS

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FRONTEND_DIR = os.path.join(BASE_DIR, "frontend")

app = Flask(
    __name__,
    template_folder=os.path.join(FRONTEND_DIR, "templates"),
    static_folder=os.path.join(FRONTEND_DIR, "static"),
)


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    payload = request.get_json(force=True)

    # basic validation: all required fields present
    missing = [col for col in FEATURE_COLUMNS if col not in payload]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        result = predict_with_explanation(payload)
    except (ValueError, TypeError) as e:
        return jsonify({"error": f"Invalid input: {e}"}), 400

    return jsonify(result)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
