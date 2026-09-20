"""Flask web application for student placement predictions."""

from flask import Flask, jsonify, render_template, request

from predict import FEATURES, predict_student


app = Flask(__name__, template_folder="../frontend/templates", static_folder="../frontend/static")


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/health")
def health():
    return jsonify({"status": "ok"})


@app.post("/predict")
def predict():
    payload = request.get_json(silent=True) or request.form.to_dict()
    missing = [feature for feature in FEATURES if feature not in payload or payload[feature] == ""]
    if missing:
        return jsonify({"error": f"Missing fields: {', '.join(missing)}"}), 400
    try:
        return jsonify(predict_student(payload))
    except (TypeError, ValueError) as error:
        return jsonify({"error": f"Please enter valid numeric values: {error}"}), 400
    except FileNotFoundError as error:
        return jsonify({"error": str(error)}), 503


if __name__ == "__main__":
    app.run(debug=True)