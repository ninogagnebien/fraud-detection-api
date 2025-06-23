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
FEATURE_NAMES = [
    'urgent_keywords', 'suspicious_phrases', 'title_length', 'description_length',
    'price', 'area_sqm', 'price_per_sqm', 'price_deviation',
    'account_age_days', 'contact_verified', 'listings_count', 'user_avg_price',
    'user_price_variability', 'user_location_diversity', 'low_account_age',
    'location_encoded', 'iso_anomaly_score', 'svm_anomaly_score',
    'reconstruction_error', 'lstm_reconstruction_error', 'is_anomaly_iso'
]

# Localisation encoding
LOCATION_MAPPING = {
    'orchard': 1, 'bukit timah': 2, 'tampines': 3, 
    'hougang': 4, 'toa payoh': 5, 'jurong': 6
}

try:
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    model = None

def clean_text(text):
    if text is None:
        return ""
    text = re.sub(r'[^\w\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text)
    return text.lower().strip()

def extract_features(listing_data):
    # 1. Text features
    title = clean_text(listing_data.get('title', ''))
    description = clean_text(listing_data.get('description', ''))
    urgent_keywords = sum(1 for word in ['urgent', 'limited', 'act fast', 'cheap', 'deal ends', 'quick'] if word in title)
    suspicious_phrases = sum(1 for phrase in ['no viewing', 'pay deposit', 'direct owner', 'no agent', 'fake listings beware'] if phrase in description)
    title_length = len(title)
    description_length = len(description)
    
    # 2. Numeric features
    price = float(listing_data.get('price', 0) or 0)
    area_sqm = float(listing_data.get('area_sqm', 0) or 0)
    price_per_sqm = float(listing_data.get('price_per_sqm', 0) or (price/area_sqm if area_sqm else 0))
    
    # 3. Price deviation (pas calculable sans stats globales, donc 0)
    price_deviation = 0
    
    # 4. User/account features (non dispo en live, donc 0)
    account_age_days = int(listing_data.get('account_age_days', 0) or 0)
    contact_verified = int(bool(listing_data.get('contact_verified', False)))
    listings_count = 0
    user_avg_price = 0
    user_price_variability = 0
    user_location_diversity = 0

    # 5. Derived features
    low_account_age = int(account_age_days < 152)
    location = str(listing_data.get('location', '')).lower()
    location_encoded = LOCATION_MAPPING.get(location, 0)
    
    # 6. Anomaly features (non dispo en live, donc 0)
    iso_anomaly_score = 0
    svm_anomaly_score = 0
    reconstruction_error = 0
    lstm_reconstruction_error = 0
    is_anomaly_iso = 0

    # Respecte l'ordre exact des features
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
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_fraud():
    default_response = {
        'is_fraud': False,
        'fraud_probability': 0.0,
        'confidence_level': 'Non défini',
        'risk_factors': [],
        'recommendation': 'PRÉDICTION_ÉCHOUÉE'
    }
    try:
        if not model:
            return jsonify({**default_response, 'error': 'Modèle non disponible'}), 500
        data = request.get_json()
        if not data:
            return jsonify({**default_response, 'error': 'Aucune donnée reçue'}), 400
        # Préparation des features
        features = extract_features(data)
        # Prédiction
        probability = float(model.predict_proba(features)[0][1])
        is_fraud = probability > 0.5
        confidence = 'Élevée' if probability > 0.8 or probability < 0.2 else 'Moyenne'
        if probability > 0.7:
            recommendation = 'REJETER'
        elif probability > 0.3:
            recommendation = 'RÉVISION_MANUELLE'
        else:
            recommendation = 'APPROUVER'
        # Facteurs de risque (exemple simple)
        risk_factors = []
        if not data.get('contact_verified', True):
            risk_factors.append("Contact non vérifié")
        if low_account_age := int(data.get('account_age_days', 0) or 0) < 152:
            risk_factors.append("Compte récent (< 152 jours)")
        if urgent_keywords := sum(1 for word in ['urgent', 'limited', 'act fast', 'cheap', 'deal ends', 'quick'] if word in clean_text(data.get('title', ''))):
            risk_factors.append("Langage d'urgence détecté")
        if suspicious_phrases := sum(1 for phrase in ['no viewing', 'pay deposit', 'direct owner', 'no agent', 'fake listings beware'] if phrase in clean_text(data.get('description', ''))):
            risk_factors.append("Phrases suspectes dans la description")
        return jsonify({
            'is_fraud': bool(is_fraud),
            'fraud_probability': round(probability, 3),
            'confidence_level': confidence,
            'risk_factors': risk_factors,
            'recommendation': recommendation
        })
    except Exception as e:
        print(f"❌ Erreur API: {e}")
        return jsonify({**default_response, 'error': f'Erreur prédiction: {str(e)}'}), 500

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
            "name": "Annonce Frauduleuse",
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
            "name": "Annonce Légitime",
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
