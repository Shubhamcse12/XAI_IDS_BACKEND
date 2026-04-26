from flask import Blueprint, request, jsonify
import pickle
import os

api = Blueprint('api', __name__)

@api.route('/predict', methods=['POST', 'OPTIONS', 'GET'])
def predict():
    if request.method == 'OPTIONS':
        return '', 200
    if request.method == 'GET':
        return jsonify({'status': 'ok'}), 200
    try:
        data = request.get_json()
        if not data:
            return jsonify({'error': 'No input data provided'}), 400
        features = data.get('features')
        if features is None:
            return jsonify({'error': 'Missing features'}), 400
        model_path = os.path.join(os.path.dirname(__file__), '../models/RF_Train70Percent.pkl')
        if not os.path.exists(model_path):
            return jsonify({'error': 'Model file not found'}), 500
        with open(model_path, 'rb') as f:
            model = pickle.load(f)
        try:
            prediction = model.predict([features])
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'}), 500
        return jsonify({'prediction': prediction[0]})
    except Exception as e:
        return jsonify({'error': f'Internal server error: {str(e)}'}), 500
