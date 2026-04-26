from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import os
import numpy as np
import shap
import requests

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------
# GOOGLE DRIVE MODEL LINKS
# ---------------------------------------------------
RF_URL = "https://drive.google.com/uc?export=download&id=1ybZi9Lk80FXBcjm8SXy_rRM-Lg6qquRw"
SVM_URL = "https://drive.google.com/uc?export=download&id=1RSqVpd-4Gisg2H-6uEJhoP2QDGM2lTVX"

rf_model_path = os.path.join(MODELS_DIR, "rf_model_25features.pkl")
svm_model_path = os.path.join(MODELS_DIR, "svm_model_25features.pkl")


# ---------------------------------------------------
# DOWNLOAD MODELS IF NOT EXISTS
# ---------------------------------------------------
def download_file(url, path):
    if not os.path.exists(path):
        print(f"Downloading {os.path.basename(path)} ...")
        response = requests.get(url, stream=True)
        response.raise_for_status()

        with open(path, "wb") as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    f.write(chunk)

        print(f"{os.path.basename(path)} downloaded.")


download_file(RF_URL, rf_model_path)
download_file(SVM_URL, svm_model_path)

# ---------------------------------------------------
# LOAD RF MODEL
# ---------------------------------------------------
rf_data = load(rf_model_path)

if isinstance(rf_data, dict):
    rf_model = rf_data.get("model")
    rf_scaler = rf_data.get("scaler")
    rf_features = rf_data.get("features")
else:
    rf_model = rf_data
    rf_scaler = None
    rf_features = None

# ---------------------------------------------------
# LOAD SVM MODEL
# ---------------------------------------------------
svm_data = load(svm_model_path)

if isinstance(svm_data, dict):
    svm_model = svm_data.get("model")
    svm_scaler = svm_data.get("scaler")
else:
    svm_model = svm_data
    svm_scaler = None

print("✅ Models Loaded Successfully")
print("RF Expected Feature Count:", rf_model.n_features_in_)

rf_explainer = shap.TreeExplainer(rf_model)

# ---------------------------------------------------
# ROUTES
# ---------------------------------------------------

@app.route("/")
def home():
    return jsonify({"message": "XAI IDS Backend Running"})


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "Backend Running"}), 200


@app.route("/features", methods=["GET"])
def get_features():
    try:
        if hasattr(rf_model, "feature_names_in_"):
            features = rf_model.feature_names_in_.tolist()
        elif rf_features:
            features = rf_features
        else:
            return jsonify({"error": "Feature names not available"}), 500

        return jsonify({"features": features}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/importance", methods=["GET"])
def importance():
    try:
        importances = rf_model.feature_importances_

        if hasattr(rf_model, "feature_names_in_"):
            feature_names = rf_model.feature_names_in_
        else:
            feature_names = [f"Feature_{i}" for i in range(len(importances))]

        data = [
            {
                "feature": feature_names[i],
                "importance": float(importances[i])
            }
            for i in range(len(importances))
        ]

        data.sort(key=lambda x: x["importance"], reverse=True)

        return jsonify({"importance": data}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "Missing features"}), 400

        features = data["features"]

        if len(features) != rf_model.n_features_in_:
            return jsonify({
                "error": f"Expected {rf_model.n_features_in_} features, got {len(features)}"
            }), 400

        input_data = np.array([features])

        if rf_scaler is not None:
            input_data = rf_scaler.transform(input_data)

        # RF Prediction
        rf_pred = rf_model.predict(input_data)[0]
        rf_conf = (
            rf_model.predict_proba(input_data)[0].max()
            if hasattr(rf_model, "predict_proba")
            else 0.5
        )

        # SVM Prediction
        svm_input = np.array([features])

        if svm_scaler is not None:
            svm_input = svm_scaler.transform(svm_input)

        svm_pred = svm_model.predict(svm_input)[0]
        svm_conf = (
            svm_model.predict_proba(svm_input)[0].max()
            if hasattr(svm_model, "predict_proba")
            else 0.5
        )

        final_pred = 1 if (rf_pred + svm_pred) >= 1 else 0

        return jsonify({
            "rf_prediction": int(rf_pred),
            "rf_confidence": float(rf_conf),
            "svm_prediction": int(svm_pred),
            "svm_confidence": float(svm_conf),
            "final_prediction": int(final_pred)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/explain", methods=["POST"])
def explain():
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "Missing features"}), 400

        features = np.array([data["features"]])

        if rf_scaler is not None:
            features = rf_scaler.transform(features)

        rf_pred = rf_model.predict(features)[0]

        shap_values = rf_explainer(features)
        shap_array = shap_values.values

        if len(shap_array.shape) == 3:
            class_index = int(rf_pred)
            shap_class = shap_array[0, :, class_index]
            base_value = float(shap_values.base_values[0][class_index])
        else:
            shap_class = shap_array[0]
            base_value = float(shap_values.base_values[0])

        if hasattr(rf_model, "feature_names_in_"):
            feature_names = rf_model.feature_names_in_
        elif rf_features:
            feature_names = rf_features
        else:
            feature_names = [f"Feature_{i}" for i in range(len(shap_class))]

        explanation = []

        for i in range(len(shap_class)):
            shap_val = float(shap_class[i])

            explanation.append({
                "feature": feature_names[i],
                "value": float(features[0][i]),
                "shap_value": shap_val,
                "impact": "positive" if shap_val > 0 else "negative"
            })

        explanation.sort(key=lambda x: abs(x["shap_value"]), reverse=True)

        return jsonify({
            "prediction_class": int(rf_pred),
            "base_value": base_value,
            "shap_values": explanation
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)