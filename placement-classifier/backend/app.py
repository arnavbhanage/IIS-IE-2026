from pathlib import Path
import json
from flask import Flask, request, jsonify, send_from_directory
from werkzeug.exceptions import BadRequest, UnsupportedMediaType
from predict import predict_with_explanation

ROOT = Path(__file__).resolve().parents[1]
DIST = ROOT / 'frontend/dist'
app = Flask(__name__, static_folder=str(DIST / 'assets'), static_url_path='/assets')

@app.get('/')
def index():
    if not (DIST / 'index.html').exists():
        return 'Build the UI first: cd placement-classifier/frontend && npm install && npm run build', 503
    return send_from_directory(DIST, 'index.html')

@app.post('/predict')
def predict():
    try:
        return jsonify(predict_with_explanation(request.get_json()))
    except (ValueError, TypeError, BadRequest, UnsupportedMediaType) as exc:
        return jsonify(error=str(exc)), 400

@app.get('/model-info')
def model_info():
    return jsonify(json.loads((ROOT / 'model/metrics.json').read_text()))

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)
