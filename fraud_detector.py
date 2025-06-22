#!/usr/bin/env python3
"""
Fraud Detection Model - Script Principal d'Utilisation
Détection de Fraude Immobilière - Singapour

Usage:
    python fraud_detector.py --test
    python fraud_detector.py --predict "title" "description" price area_sqm account_age
"""

import joblib
import pandas as pd
import numpy as np
import argparse
import json
from datetime import datetime

class FraudDetector:
    def __init__(self, models_dir='models'):
        """Initialiser le détecteur de fraude"""
        self.models_dir = models_dir
        self.load_models()

    def load_models(self):
        """Charger tous les modèles nécessaires"""
        try:
            self.rf_model = joblib.load(f'{self.models_dir}/best_model.pkl')
            self.iso_forest = joblib.load(f'{self.models_dir}/isolation_forest.pkl')
            self.scaler = joblib.load(f'{self.models_dir}/scaler.pkl')
            self.metadata = joblib.load(f'{self.models_dir}/metadata.pkl')
            print("✅ Modèles chargés avec succès")
        except Exception as e:
            print(f"❌ Erreur lors du chargement des modèles: {e}")
            raise

    def extract_features(self, listing_data):
        """Extraire les features à partir des données d'annonce"""

        # Features textuelles
        title = str(listing_data.get('title', ''))
        description = str(listing_data.get('description', ''))

        urgent_keywords = ['urgent', 'limited', 'act fast', 'deal ends', 'quick', 'cheap', 'best price']
        suspicious_phrases = ['no viewing', 'pay deposit', 'direct owner', 'no agent', 'fake listings beware']

        features = {}
        features['urgent_keywords'] = sum(1 for keyword in urgent_keywords if keyword.lower() in title.lower())
        features['suspicious_phrases'] = sum(1 for phrase in suspicious_phrases if phrase.lower() in description.lower())
        features['title_length'] = len(title)
        features['description_length'] = len(description)
        features['title_exclamation'] = title.count('!')

        # Features numériques
        features['price'] = listing_data.get('price', 0)
        features['area_sqm'] = listing_data.get('area_sqm', 0)
        features['price_per_sqm'] = listing_data.get('price_per_sqm', 0)
        features['account_age_days'] = listing_data.get('account_age_days', 0)

        # Features dérivées
        features['price_deviation'] = 0
        features['contact_verified'] = int(listing_data.get('contact_verified', False))
        features['low_account_age'] = int(features['account_age_days'] < 152)

        # Features utilisateur (simplifiées)
        features['user_listing_count'] = listing_data.get('user_listing_count', 1)
        features['user_avg_price'] = features['price']
        features['user_price_variability'] = 0
        features['user_location_diversity'] = 1

        # Location encoding
        location_mapping = {'Tampines': 0, 'Orchard': 1, 'Hougang': 2, 'Bukit Timah': 3, 'Toa Payoh': 4, 'Jurong': 5}
        features['location_encoded'] = location_mapping.get(listing_data.get('location', 'Tampines'), 0)

        # Features de prix
        features['price_too_low'] = int(features['price_per_sqm'] < 20)
        features['price_very_high'] = int(features['price_per_sqm'] > 55)

        return features

    def predict(self, listing_data):
        """Prédire si une annonce est frauduleuse"""

        # 1. Extraire les features de base
        features = self.extract_features(listing_data)
        base_features = pd.DataFrame([features])

        # 2. Générer les features d'anomalies
        X_scaled = self.scaler.transform(base_features[self.metadata['feature_names']])
        iso_scores_prob = self.iso_forest.decision_function(X_scaled)
        iso_binary = (self.iso_forest.predict(X_scaled) == -1).astype(int)

        # 3. Combiner toutes les features
        anomaly_features = pd.DataFrame({
            'iso_anomaly_score': iso_scores_prob,
            'is_anomaly_iso': iso_binary
        })

        final_features = pd.concat([base_features, anomaly_features], axis=1)

        # 4. Prédiction finale
        fraud_prob = self.rf_model.predict_proba(final_features)[0][1]
        prediction = "FRAUDE" if fraud_prob > 0.5 else "LÉGITIME"

        return {
            'prediction': prediction,
            'fraud_probability': fraud_prob,
            'confidence': 'ÉLEVÉE' if fraud_prob > 0.8 or fraud_prob < 0.2 else 'MOYENNE',
            'timestamp': datetime.now().isoformat(),
            'risk_factors': {
                'urgent_keywords': features['urgent_keywords'],
                'suspicious_phrases': features['suspicious_phrases'],
                'low_account_age': bool(features['low_account_age']),
                'contact_verified': bool(features['contact_verified']),
                'price_anomaly': bool(features['price_too_low'] or features['price_very_high']),
                'anomaly_score': float(iso_scores_prob[0])
            }
        }

    def batch_predict(self, listings):
        """Prédire pour plusieurs annonces"""
        results = []
        for i, listing in enumerate(listings):
            try:
                result = self.predict(listing)
                result['listing_id'] = listing.get('listing_id', f'listing_{i+1}')
                results.append(result)
            except Exception as e:
                results.append({
                    'listing_id': listing.get('listing_id', f'listing_{i+1}'),
                    'error': str(e),
                    'prediction': 'ERROR'
                })
        return results

def run_tests():
    """Exécuter des tests de validation"""
    print("🔬 TESTS DE VALIDATION DU MODÈLE")
    print("=" * 50)

    detector = FraudDetector()

    test_cases = [
        {
            'name': 'Listing Suspect',
            'data': {
                'title': 'Urgent Rental: Tampines Only Today! Cheap deal!',
                'description': 'Direct from owner. No agent fee! Pay deposit to secure. No viewing needed.',
                'price': 1000,
                'area_sqm': 50,
                'price_per_sqm': 20,
                'account_age_days': 25,
                'contact_verified': False,
                'location': 'Tampines'
            }
        },
        {
            'name': 'Listing Légitime',
            'data': {
                'title': 'Modern 75 sqm Apartment in Orchard',
                'description': 'Newly renovated unit with great ventilation and natural lighting.',
                'price': 2500,
                'area_sqm': 75,
                'price_per_sqm': 33,
                'account_age_days': 800,
                'contact_verified': True,
                'location': 'Orchard'
            }
        }
    ]

    for test_case in test_cases:
        print(f"\n🧪 {test_case['name']}:")
        result = detector.predict(test_case['data'])
        print(f"  Prédiction: {result['prediction']}")
        print(f"  Probabilité: {result['fraud_probability']:.2%}")
        print(f"  Confiance: {result['confidence']}")
        print("  Facteurs de risque principaux:")
        for factor, value in result['risk_factors'].items():
            if (isinstance(value, bool) and value) or (isinstance(value, (int, float)) and value > 0):
                print(f"    ⚠️  {factor}: {value}")

def main():
    parser = argparse.ArgumentParser(description='Détecteur de Fraude Immobilière')
    parser.add_argument('--test', action='store_true', help='Exécuter les tests de validation')
    parser.add_argument('--predict', nargs=5, metavar=('title', 'description', 'price', 'area_sqm', 'account_age'),
                       help='Prédire pour une annonce spécifique')
    parser.add_argument('--batch', type=str, help='Fichier JSON avec plusieurs annonces')

    args = parser.parse_args()

    if args.test:
        run_tests()
    elif args.predict:
        title, description, price, area_sqm, account_age = args.predict

        listing_data = {
            'title': title,
            'description': description,
            'price': float(price),
            'area_sqm': float(area_sqm),
            'price_per_sqm': float(price) / float(area_sqm),
            'account_age_days': int(account_age),
            'contact_verified': False,
            'location': 'Tampines'
        }

        detector = FraudDetector()
        result = detector.predict(listing_data)

        print("\n🔍 RÉSULTAT DE LA PRÉDICTION:")
        print("=" * 40)
        print(f"Prédiction: {result['prediction']}")
        print(f"Probabilité de fraude: {result['fraud_probability']:.2%}")
        print(f"Niveau de confiance: {result['confidence']}")
        print("\nFacteurs de risque détectés:")
        for factor, value in result['risk_factors'].items():
            print(f"  {factor}: {value}")

    elif args.batch:
        with open(args.batch, 'r') as f:
            listings = json.load(f)

        detector = FraudDetector()
        results = detector.batch_predict(listings)

        print(f"\n📊 RÉSULTATS BATCH ({len(results)} annonces):")
        for result in results:
            print(f"  {result.get('listing_id', 'N/A')}: {result['prediction']} ({result.get('fraud_probability', 0):.2%})")

    else:
        print("Usage: python fraud_detector.py --test")
        print("       python fraud_detector.py --predict 'titre' 'description' prix surface_m2 age_compte")

if __name__ == '__main__':
    main()
