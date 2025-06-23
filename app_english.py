from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import numpy as np
import os
import re

app = Flask(__name__)
CORS(app)
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

MODEL_PATH = 'models/fraud_detection_model.pkl'

# Mapping for location encoding
LOCATION_MAPPING = {
    'orchard': 1, 'bukit timah': 2, 'tampines': 3, 
    'hougang': 4, 'toa payoh': 5, 'jurong': 6
}

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Model loaded successfully")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    model = None

def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def extract_features(listing_data):
    title = clean_text(listing_data.get('title', ''))
    description = clean_text(listing_data.get('description', ''))
    urgent_keywords = sum(1 for word in ['urgent', 'limited', 'act fast', 'cheap', 'deal ends', 'quick'] if word in title)
    suspicious_phrases = sum(1 for phrase in ['no viewing', 'pay deposit', 'direct owner', 'no agent', 'fake listings beware'] if phrase in description)
    title_length = len(title)
    description_length = len(description)
    
    price = float(listing_data.get('price', 0) or 0)
    area_sqm = float(listing_data.get('area_sqm', 0) or 0)
    price_per_sqm = float(listing_data.get('price_per_sqm', 0) or (price / area_sqm if area_sqm else 0))
    
    price_deviation = 0  # placeholder
    account_age_days = int(listing_data.get('account_age_days', 0) or 0)
    contact_verified = int(bool(listing_data.get('contact_verified', False)))
    listings_count = 0
    user_avg_price = 0
    user_price_variability = 0
    user_location_diversity = 0
    low_account_age = int(account_age_days < 152)
    location = str(listing_data.get('location', '')).lower()
    location_encoded = LOCATION_MAPPING.get(location, 0)
    
    iso_anomaly_score = 0
    svm_anomaly_score = 0
    reconstruction_error = 0
    lstm_reconstruction_error = 0
    is_anomaly_iso = 0

    features = [
        urgent_keywords, suspicious_phrases, title_length, description_length,
        price, area_sqm, price_per_sqm, price_deviation,
        account_age_days, contact_verified, listings_count, user_avg_price,
        user_price_variability, user_location_diversity, low_account_age,
        location_encoded, iso_anomaly_score, svm_anomaly_score,
        reconstruction_error, lstm_reconstruction_error, is_anomaly_iso
    ]
    return np.array(features).reshape(1, -1)

@app.route('/')
def home():
    return render_template('index_en.html')

@app.route('/api/predict', methods=['POST'])
def predict_fraud():
    default_response = {
        'is_fraud': False,
        'fraud_probability': 0.0,
        'confidence_level': 'Undefined',
        'risk_factors': [],
        'recommendation': 'PREDICTION_FAILED'
    }
    try:
        if not model:
            return jsonify({**default_response, 'error': 'Model not available'}), 500
        data = request.get_json()
        if not data:
            return jsonify({**default_response, 'error': 'No data received'}), 400
        
        features = extract_features(data)
        probability = float(model.predict_proba(features)[0][1])
        is_fraud = probability > 0.5
        confidence = 'High' if probability > 0.8 or probability < 0.2 else 'Medium'
        if probability > 0.7:
            recommendation = 'REJECT'
        elif probability > 0.3:
            recommendation = 'MANUAL_REVIEW'
        else:
            recommendation = 'APPROVE'

        risk_factors = []
        if not data.get('contact_verified', True):
            risk_factors.append("Unverified contact")
        if (int(data.get('account_age_days', 0) or 0)) < 152:
            risk_factors.append("Recent account (< 152 days)")
        if sum(1 for word in ['urgent', 'limited', 'act fast', 'cheap', 'deal ends', 'quick'] if word in clean_text(data.get('title', ''))):
            risk_factors.append("Urgent language detected")
        if sum(1 for phrase in ['no viewing', 'pay deposit', 'direct owner', 'no agent', 'fake listings beware'] if phrase in clean_text(data.get('description', ''))):
            risk_factors.append("Suspicious phrases detected")

        return jsonify({
            'is_fraud': bool(is_fraud),
            'fraud_probability': round(probability, 3),
            'confidence_level': confidence,
            'risk_factors': risk_factors,
            'recommendation': recommendation
        })
    except Exception as e:
        print(f"❌ API error: {e}")
        return jsonify({**default_response, 'error': f'Prediction error: {str(e)}'}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/api/examples', methods=['GET'])
def get_examples():
    examples = [
        {
            "name": "Fraudulent Listing",
            "data": {
                "title": "Urgent! Cheap apartment in Orchard - Act Fast!",
                "description": "No viewing needed, pay deposit to secure. Direct from owner, no agent fee!",
                "price": 800,
                "area_sqm": 50,
                "price_per_sqm": 16,
                "location": "Orchard",
                "contact_verified": False,
                "account_age_days": 15
            }
        },
        {
            "name": "Legitimate Listing",
            "data": {
                "title": "Beautiful 2-bedroom apartment in Marina Bay",
                "description": "Well-maintained unit with modern amenities. Viewing available on weekends.",
                "price": 4500,
                "area_sqm": 85,
                "price_per_sqm": 53,
                "location": "Orchard",
                "contact_verified": True,
                "account_age_days": 850
            }
        }
    ]
    return jsonify(examples)

@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
