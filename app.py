from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)  # Permettre les requêtes cross-origin

# Charger le modèle au démarrage
MODEL_PATH = 'models/fraud_model.pkl'
try:
    model = joblib.load(MODEL_PATH)
    print("✅ Modèle chargé avec succès")
except Exception as e:
    print(f"❌ Erreur chargement modèle: {e}")
    model = None

class FraudPredictor:
    def __init__(self, model):
        self.model = model
    
    def extract_features(self, listing_data):
        """Extraction des features à partir des données d'annonce"""
        features = {}
        
        # Features textuelles
        title = listing_data.get('title', '').lower()
        description = listing_data.get('description', '').lower()
        
        # Mots-clés urgents
        urgent_keywords = ['urgent', 'limited', 'act fast', 'cheap', 'deal ends', 'quick']
        features['urgent_keywords'] = sum(1 for word in urgent_keywords if word in title)
        
        # Phrases suspectes
        suspicious_phrases = ['no viewing', 'pay deposit', 'direct owner', 'no agent', 'fake listings beware']
        features['suspicious_phrases'] = sum(1 for phrase in suspicious_phrases if phrase in description)
        
        # Longueurs de texte
        features['title_length'] = len(title)
        features['description_length'] = len(description)
        
        # Features numériques
        features['price'] = float(listing_data.get('price', 0))
        features['area_sqm'] = float(listing_data.get('area_sqm', 0))
        features['price_per_sqm'] = float(listing_data.get('price_per_sqm', 0))
        features['account_age_days'] = int(listing_data.get('account_age_days', 365))
        features['contact_verified'] = int(listing_data.get('contact_verified', False))
        
        # Features dérivées
        features['low_account_age'] = 1 if features['account_age_days'] < 152 else 0
        features['very_low_price'] = 1 if features['price_per_sqm'] < 20 else 0
        features['very_high_price'] = 1 if features['price_per_sqm'] > 55 else 0
        
        # Localisation (encoding simple)
        location_mapping = {'orchard': 1, 'bukit timah': 2, 'tampines': 3, 'hougang': 4, 'toa payoh': 5, 'jurong': 6}
        location = listing_data.get('location', '').lower()
        features['location_encoded'] = location_mapping.get(location, 0)
        
        return list(features.values())
    
    def predict(self, listing_data):
        """Prédiction de fraude"""
        if not self.model:
            return {'error': 'Modèle non disponible'}
        
        try:
            features = self.extract_features(listing_data)
            features_array = np.array(features).reshape(1, -1)
            
            # Prédiction
            if hasattr(self.model, 'predict_proba'):
                probability = self.model.predict_proba(features_array)[0][1]
            else:
                # Fallback si pas de predict_proba
                prediction = self.model.predict(features_array)[0]
                probability = float(prediction)
            
            is_fraud = probability > 0.5
            confidence = 'Élevée' if probability > 0.8 or probability < 0.2 else 'Moyenne'
            
            # Facteurs de risque identifiés
            risk_factors = self.get_risk_factors(listing_data, features)
            
            return {
                'is_fraud': bool(is_fraud),
                'fraud_probability': round(float(probability), 3),
                'confidence_level': confidence,
                'risk_factors': risk_factors,
                'recommendation': 'REJETER' if probability > 0.7 else 'RÉVISION MANUELLE' if probability > 0.3 else 'APPROUVER'
            }
            
        except Exception as e:
            return {'error': f'Erreur prédiction: {str(e)}'}
    
    def get_risk_factors(self, listing_data, features):
        """Identification des facteurs de risque"""
        risks = []
        
        if features[10]:  # low_account_age
            risks.append("Compte récent (< 152 jours)")
        
        if not listing_data.get('contact_verified', True):
            risks.append("Contact non vérifié")
        
        if features[0] > 0:  # urgent_keywords
            risks.append("Langage d'urgence détecté")
        
        if features[1] > 0:  # suspicious_phrases
            risks.append("Phrases suspectes dans la description")
        
        if features[11]:  # very_low_price
            risks.append("Prix anormalement bas")
        
        if features[12]:  # very_high_price
            risks.append("Prix anormalement élevé")
        
        return risks

# Initialiser le prédicteur
predictor = FraudPredictor(model) if model else None

@app.route('/')
def home():
    """Page d'accueil avec interface de test"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_fraud():
    """Endpoint de prédiction de fraude"""
    try:
        if not predictor:
            return jsonify({'error': 'Modèle non disponible'}), 500
        
        data = request.json
        if not data:
            return jsonify({'error': 'Données manquantes'}), 400
        
        # Validation des champs requis
        required_fields = ['title', 'description', 'price', 'area_sqm', 'account_age_days']
        for field in required_fields:
            if field not in data:
                return jsonify({'error': f'Champ requis manquant: {field}'}), 400
        
        # Prédiction
        result = predictor.predict(data)
        
        # Log de la prédiction
        print(f"🔍 Prédiction: {data.get('title', '')[:50]}... -> {result.get('fraud_probability', 0):.3f}")
        
        return jsonify(result)
        
    except Exception as e:
        print(f"❌ Erreur API: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérification de l'état de l'API"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'timestamp': datetime.now().isoformat()
    })

@app.route('/api/examples', methods=['GET'])
def get_examples():
    """Exemples d'annonces pour les tests"""
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

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
