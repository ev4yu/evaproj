from flask import Flask, request, jsonify, send_from_directory
import joblib
import os

import joblib

# After training

app = Flask(__name__, static_folder='static')

model = joblib.load("ok.pkl")  # Make sure model.pkl is in root folder
# joblib.dump(model, "best_xgb_model.pkl")

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()
    features = data.get("features") if data else None
    if features is None:
        return jsonify({"error": "Missing 'features' in request body"}), 400
    prediction = model.predict([features])
    return jsonify({"prediction": prediction.tolist()})

@app.route("/", methods=["GET"])
def serve_index():
    return send_from_directory("static", "index.html")

@app.route("/model", methods=["GET"])
def serve_model():
    return send_from_directory("static", "model.html")

if __name__ == "__main__":
    app.run(debug=True)