from flask import Flask, request, jsonify, send_from_directory
import xgboost as xgb
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler 

app = Flask(__name__, static_folder='static')

# Load the model
model = xgb.XGBClassifier()
model.load_model('models/xgboost_fetal_anomalies_model.json')

def load_scaler(path='models/scaler.pkl'):
    with open(path, 'rb') as f:
        scaler_data = joblib.load(f)
        
    scaler = StandardScaler()

    scaler.mean_ = scaler_data['mean_']
    scaler.scale_ = scaler_data['scale_']

    return scaler

prediction_mapping = {0: "Normal", 1: "Suspect", 2: "Pathological"}

scaler = load_scaler()

confidence_threshold = 0.8  

# Route to serve the HTML form
@app.route('/')
def serve_html():
    return send_from_directory(app.static_folder, 'index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    features = np.array(data['features']).astype(float)
    print(features.shape)
        # Ensure the features array is 2D (number of samples, number of features)
    if features.ndim == 3:
        features = features.reshape(features.shape[1], features.shape[2])
    print("------------------")
    print(features.shape)
    # Scale the features using the loaded scaler
    if hasattr(scaler, 'transform'): 
        scaled_features = scaler.transform(features)
    else:
        return jsonify({'error': 'Scaler object is not correctly loaded'}), 500

    # Model prediction and probability calculation
    probabilities = model.predict_proba(scaled_features)  # Get prediction probabilities
    prediction = np.argmax(probabilities, axis=1)  # Get the class with the highest probability

    # Calculate confidence score for the predicted class
    confidence_score = np.max(probabilities, axis=1)

    # Map numeric predictions to their respective string labels
    mapped_prediction = [prediction_mapping[p] for p in prediction]
    
    # Prepare the response
    response = {
        'prediction': mapped_prediction[0],
        'confidence': float(confidence_score[0])
    }

    # Check if the confidence score is below the threshold
    if confidence_score[0] < confidence_threshold:
        response['warning'] = "The model is not very confident about this prediction. Please review the data carefully."

    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
