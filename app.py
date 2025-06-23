from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import os
import re
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Configuration pour forcer le rechargement des fichiers statiques
app.config['SEND_FILE_MAX_AGE_DEFAULT'] = 0

# Charger le modèle au démarrage
MODEL_PATH = 'models/fraud_detection_model.pkl'
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
        try:
            title = str(listing_data.get('title', '')).lower()
            description = str(listing_data.get('description', '')).lower()
            
            # 1. Text features
            urgent_keywords = sum(1 for word in ['urgent', 'limited', 'act fast', 'cheap', 'deal ends', 'quick'] if word in title)
            suspicious_phrases = sum(1 for phrase in ['no viewing', 'pay deposit', 'direct owner', 'no agent', 'fake listings beware'] if phrase in description)
            title_length = len(title)
            description_length = len(description)

            # 2. Numeric features
            price = float(listing_data.get('price', 0) or 0)
            area_sqm = float(listing_data.get('area_sqm', 0) or 0)
            price_per_sqm = float(listing_data.get('price_per_sqm', 0) or (price / area_sqm if area_sqm else 0))
            price_deviation = 0  # Can't compute in production
            account_age_days = int(listing_data.get('account_age_days', 0) or 0)
            contact_verified = int(bool(listing_data.get('contact_verified', False)))

            # 3. User-related features – placeholder values in production
            listings_count = 0
            user_avg_price = 0
            user_price_variability = 0
            user_location_diversity = 0

            # 4. Derived features
            low_account_age = int(account_age_days < 152)
            location = str(listing_data.get('location', '')).lower()
            location_encoded = {
                'orchard': 1, 'bukit timah': 2, 'tampines': 3, 
                'hougang': 4, 'toa payoh': 5, 'jurong': 6
            }.get(location, 0)

            # 5. Anomaly detection placeholders
            iso_anomaly_score = 0
            svm_anomaly_score = 0
            reconstruction_error = 0
            lstm_reconstruction_error = 0
            is_anomaly_iso = 0

            return np.array([
                urgent_keywords, suspicious_phrases, title_length, description_length,
                price, area_sqm, price_per_sqm, price_deviation,
                account_age_days, contact_verified, listings_count, user_avg_price,
                user_price_variability, user_location_diversity, low_account_age,
                location_encoded, iso_anomaly_score, svm_anomaly_score,
                reconstruction_error, lstm_reconstruction_error, is_anomaly_iso
            ])
        except Exception as e:
            print(f"Erreur extract_features : {e}")
            return np.zeros(21)

    # def extract_features(self, listing_data):
    #     """Extraction des features à partir des données d'annonce"""
    #     try:
    #         features = {}
            
    #         # Sécuriser les accès aux données
    #         title = str(listing_data.get('title', '')).lower()
    #         description = str(listing_data.get('description', '')).lower()
            
    #         # Features textuelles
    #         urgent_keywords = ['urgent', 'limited', 'act fast', 'cheap', 'deal ends', 'quick']
    #         features['urgent_keywords'] = sum(1 for word in urgent_keywords if word in title)
            
    #         suspicious_phrases = ['no viewing', 'pay deposit', 'direct owner', 'no agent', 'fake listings beware']
    #         features['suspicious_phrases'] = sum(1 for phrase in suspicious_phrases if phrase in description)
            
    #         # Longueurs de texte
    #         features['title_length'] = len(title)
    #         features['description_length'] = len(description)
            
    #         # Features numériques avec validation
    #         try:
    #             features['price'] = float(listing_data.get('price', 0))
    #             features['area_sqm'] = float(listing_data.get('area_sqm', 0))
    #             features['price_per_sqm'] = float(listing_data.get('price_per_sqm', 0))
    #             features['account_age_days'] = int(listing_data.get('account_age_days', 365))
    #         except (ValueError, TypeError):
    #             features['price'] = 0.0
    #             features['area_sqm'] = 0.0
    #             features['price_per_sqm'] = 0.0
    #             features['account_age_days'] = 365
            
    #         features['contact_verified'] = int(bool(listing_data.get('contact_verified', False)))
            
    #         # Features dérivées
    #         features['low_account_age'] = 1 if features['account_age_days'] < 152 else 0
    #         features['very_low_price'] = 1 if features['price_per_sqm'] < 20 and features['price_per_sqm'] > 0 else 0
    #         features['very_high_price'] = 1 if features['price_per_sqm'] > 55 else 0
            
    #         # Localisation avec gestion d'erreur
    #         location_mapping = {
    #             'orchard': 1, 'bukit timah': 2, 'tampines': 3, 
    #             'hougang': 4, 'toa payoh': 5, 'jurong': 6
    #         }
    #         location = str(listing_data.get('location', '')).lower()
    #         features['location_encoded'] = location_mapping.get(location, 0)
            
    #         return list(features.values())
            
    #     except Exception as e:
    #         print(f"Erreur extraction features: {e}")
    #         return [0] * 12
    
    def predict(self, listing_data):
        """Prédiction de fraude avec gestion complète des erreurs"""
        default_response = {
            'is_fraud': False,
            'fraud_probability': 0.0,
            'confidence_level': 'Non défini',
            'risk_factors': [],
            'recommendation': 'ERREUR'
        }
        
        if not self.model:
            return {
                **default_response,
                'error': 'Modèle non disponible',
                'recommendation': 'MODÈLE_INDISPONIBLE'
            }
        
        try:
            if not listing_data:
                return {
                    **default_response,
                    'error': 'Données manquantes',
                    'recommendation': 'DONNÉES_MANQUANTES'
                }
            
            features = self.extract_features(listing_data)
            if len(features) == 0 or features is None or features.size == 0:
                return {
                    **default_response,
                    'error': 'Impossible d\'extraire les features',
                    'recommendation': 'EXTRACTION_ÉCHOUÉE'
                }
            
            features_array = np.array(features).reshape(1, -1)
            
            try:
                if hasattr(self.model, 'predict_proba'):
                    probability = float(self.model.predict_proba(features_array)[0][1])
                else:
                    prediction = self.model.predict(features_array)[0]
                    probability = float(prediction)
            except Exception as pred_error:
                print(f"Erreur prédiction: {pred_error}")
                return {
                    **default_response,
                    'error': f'Erreur prédiction: {str(pred_error)}',
                    'recommendation': 'PRÉDICTION_ÉCHOUÉE'
                }
            
            probability = max(0.0, min(1.0, probability))
            is_fraud = probability > 0.5
            
            if probability > 0.8 or probability < 0.2:
                confidence = 'Élevée'
            else:
                confidence = 'Moyenne'
            
            if probability > 0.7:
                recommendation = 'REJETER'
            elif probability > 0.3:
                recommendation = 'RÉVISION_MANUELLE'
            else:
                recommendation = 'APPROUVER'
            
            risk_factors = self.get_risk_factors(listing_data, features)
            
            return {
                'is_fraud': bool(is_fraud),
                'fraud_probability': round(probability, 3),
                'confidence_level': confidence,
                'risk_factors': risk_factors,
                'recommendation': recommendation
            }
            
        except Exception as e:
            print(f"Erreur générale prédiction: {e}")
            return {
                **default_response,
                'error': f'Erreur générale: {str(e)}',
                'recommendation': 'ERREUR_GÉNÉRALE'
            }
    
    def get_risk_factors(self, listing_data, features):
        """Identification des facteurs de risque"""
        risks = []
        
        try:
            account_age = int(listing_data.get('account_age_days', 365))
            if account_age < 152:
                risks.append("Compte récent (< 152 jours)")
            
            if not listing_data.get('contact_verified', True):
                risks.append("Contact non vérifié")
            
            if len(features) >= 12:
                if features[0] > 0:
                    risks.append("Langage d'urgence détecté")
                
                if features[1] > 0:
                    risks.append("Phrases suspectes dans la description")
                
                if features[10] > 0:
                    risks.append("Prix anormalement bas")
                
                if features[11] > 0:
                    risks.append("Prix anormalement élevé")
        
        except Exception as e:
            print(f"Erreur calcul facteurs de risque: {e}")
            risks.append("Erreur analyse des facteurs de risque")
        
        return risks

predictor = FraudPredictor(model) if model else None

@app.route('/')
def home():
    """Page d'accueil avec interface de test"""
    return render_template('index.html')

@app.route('/api/predict', methods=['POST'])
def predict_fraud():
    """Endpoint de prédiction de fraude avec gestion complète des erreurs"""
    default_response = {
        'is_fraud': False,
        'fraud_probability': 0.0,
        'confidence_level': 'Non défini',
        'risk_factors': [],
        'recommendation': 'ERREUR'
    }
    
    try:
        if not predictor:
            response = {
                **default_response,
                'error': 'Modèle non disponible',
                'recommendation': 'SERVICE_INDISPONIBLE'
            }
            return jsonify(response), 500
        
        data = request.get_json()
        if not data:
            response = {
                **default_response,
                'error': 'Aucune donnée reçue',
                'recommendation': 'DONNÉES_MANQUANTES'
            }
            return jsonify(response), 400
        
        required_fields = ['title', 'description', 'price', 'area_sqm', 'account_age_days']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            response = {
                **default_response,
                'error': f'Champs manquants: {", ".join(missing_fields)}',
                'recommendation': 'CHAMPS_MANQUANTS'
            }
            return jsonify(response), 400
        
        result = predictor.predict(data)
        
        required_response_fields = ['is_fraud', 'fraud_probability', 'confidence_level', 'risk_factors', 'recommendation']
        for field in required_response_fields:
            if field not in result:
                result[field] = default_response[field]
        
        title_preview = str(data.get('title', ''))[:50]
        print(f"🔍 Prédiction: {title_preview}... -> {result.get('fraud_probability', 0):.3f}")
        
        return jsonify(result), 200
        
    except Exception as e:
        print(f"❌ Erreur API: {e}")
        response = {
            **default_response,
            'error': f'Erreur serveur: {str(e)}',
            'recommendation': 'ERREUR_SERVEUR'
        }
        return jsonify(response), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    """Vérification de l'état de l'API"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None,
        'predictor_ready': predictor is not None,
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

# Configuration pour désactiver le cache en développement
@app.after_request
def add_header(response):
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    return response

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
