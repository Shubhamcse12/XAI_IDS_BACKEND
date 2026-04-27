from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load
import os
import numpy as np
import shap
import gdown

app = Flask(__name__)
CORS(app)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
MODELS_DIR = os.path.join(ROOT_DIR, "models")

os.makedirs(MODELS_DIR, exist_ok=True)

# ---------------------------------------------------
# ONLY THE 25-FEATURE MODELS (match your frontend)
# ---------------------------------------------------
MODELS = {
    "rf_model_25features.pkl":  "1ybZi9Lk80FXBcjm8SXy_rRM-Lg6qquRw",
    "svm_model_25features.pkl": "1RSqVpd-4Gisg2H-6uEJhoP2QDGM2lTVX",
}


# ---------------------------------------------------
# DOWNLOAD USING GDOWN
# ---------------------------------------------------
def download_model(file_id, path):
    if not os.path.exists(path):
        filename = os.path.basename(path)
        print(f"Downloading {filename} ...")
        gdown.download(id=file_id, output=path, quiet=False)

        with open(path, "rb") as f:
            header = f.read(15)
        if header.startswith(b"<!DOCTYPE") or header.startswith(b"<html"):
            os.remove(path)
            raise RuntimeError(
                f"Got HTML instead of model file for '{filename}'. "
                f"Make sure it is shared as 'Anyone with the link' on Google Drive."
            )
        print(f"✅ {filename} downloaded.")
    else:
        print(f"✅ {os.path.basename(path)} already exists, skipping.")


for filename, file_id in MODELS.items():
    download_model(file_id, os.path.join(MODELS_DIR, filename))


# ---------------------------------------------------
# LOAD RF MODEL
# ---------------------------------------------------
rf_data = load(os.path.join(MODELS_DIR, "rf_model_25features.pkl"))
if isinstance(rf_data, dict):
    rf_model    = rf_data.get("model")
    rf_scaler   = rf_data.get("scaler")
    rf_features = rf_data.get("features")
else:
    rf_model    = rf_data
    rf_scaler   = None
    rf_features = None


# ---------------------------------------------------
# LOAD SVM MODEL
# ---------------------------------------------------
svm_data = load(os.path.join(MODELS_DIR, "svm_model_25features.pkl"))
if isinstance(svm_data, dict):
    svm_model  = svm_data.get("model")
    svm_scaler = svm_data.get("scaler")
else:
    svm_model  = svm_data
    svm_scaler = None


print("✅ Models Loaded Successfully")
print(f"RF  - Expected Features: {rf_model.n_features_in_}")
print(f"SVM - Expected Features: {svm_model.n_features_in_}")

# SHAP explainer
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
        feature_names = (
            rf_model.feature_names_in_
            if hasattr(rf_model, "feature_names_in_")
            else [f"Feature_{i}" for i in range(len(importances))]
        )

        data = sorted(
            [
                {"feature": str(feature_names[i]), "importance": float(importances[i])}
                for i in range(len(importances))
            ],
            key=lambda x: x["importance"],
            reverse=True,
        )
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

        input_arr = np.array([features], dtype=float)

        # --- RF Prediction ---
        rf_input = rf_scaler.transform(input_arr) if rf_scaler else input_arr
        rf_pred  = int(rf_model.predict(rf_input)[0])
        rf_conf  = (
            float(rf_model.predict_proba(rf_input)[0].max())
            if hasattr(rf_model, "predict_proba") else 0.5
        )

        # --- SVM Prediction ---
        svm_input = svm_scaler.transform(input_arr) if svm_scaler else input_arr
        svm_pred  = int(svm_model.predict(svm_input)[0])
        svm_conf  = (
            float(svm_model.predict_proba(svm_input)[0].max())
            if hasattr(svm_model, "predict_proba") else 0.5
        )

        # --- Ensemble: attack if either model says attack ---
        final_pred = 1 if (rf_pred + svm_pred) >= 1 else 0

        return jsonify({
            "rf_prediction":    rf_pred,
            "rf_confidence":    rf_conf,
            "svm_prediction":   svm_pred,
            "svm_confidence":   svm_conf,
            "final_prediction": final_pred,
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/explain", methods=["POST"])
def explain():
    try:
        data = request.get_json()

        if not data or "features" not in data:
            return jsonify({"error": "Missing features"}), 400

        features  = np.array([data["features"]], dtype=float)

        if features.shape[1] != rf_model.n_features_in_:
            return jsonify({
                "error": f"Expected {rf_model.n_features_in_} features, got {features.shape[1]}"
            }), 400

        rf_input = rf_scaler.transform(features) if rf_scaler else features
        rf_pred  = int(rf_model.predict(rf_input)[0])

        shap_values = rf_explainer(rf_input)
        shap_array  = shap_values.values

        if len(shap_array.shape) == 3:
            class_index = rf_pred
            shap_class  = shap_array[0, :, class_index]
            base_value  = float(shap_values.base_values[0][class_index])
        else:
            shap_class = shap_array[0]
            base_value = float(shap_values.base_values[0])

        feature_names = (
            rf_model.feature_names_in_ if hasattr(rf_model, "feature_names_in_")
            else rf_features if rf_features
            else [f"Feature_{i}" for i in range(len(shap_class))]
        )

        explanation = sorted(
            [
                {
                    "feature":    str(feature_names[i]),
                    "value":      float(rf_input[0][i]),
                    "shap_value": float(shap_class[i]),
                    "impact":     "positive" if float(shap_class[i]) > 0 else "negative",
                }
                for i in range(len(shap_class))
            ],
            key=lambda x: abs(x["shap_value"]),
            reverse=True,
        )

        return jsonify({
            "prediction_class": rf_pred,
            "base_value":       base_value,
            "shap_values":      explanation,
        }), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)