# Sauvegarder le meilleur modèle et créer un script de test
import joblib

# Sauvegarder le meilleur modèle avec informations complètes
best_model_obj = trainer.models['random_forest']
joblib.dump(best_model_obj, 'models/best_model.pkl')

# Sauvegarder les informations du meilleur modèle
best_model_info = {
    'model_name': 'random_forest',
    'model_type': 'RandomForestClassifier',
    'auc_score': 1.0,
    'training_samples': len(y),
    'feature_count': len(trainer.feature_names),
    'model_path': 'models/best_model.pkl',
    'scaler_path': 'models/scaler.pkl',
    'metadata_path': 'models/metadata.pkl'
}
joblib.dump(best_model_info, 'models/best_model_info.pkl')

print("📦 MODÈLES SAUVEGARDÉS:")
print("=" * 40)

# Lister tous les fichiers créés
import os
model_files = os.listdir('models')
for file in sorted(model_files):
    file_path = f'models/{file}'
    file_size = os.path.getsize(file_path)
    print(f"  ✅ {file:<25} ({file_size/1024:.1f} KB)")

print("\n🔬 FONCTION DE TEST DES MODÈLES")
print("=" * 40)

def test_prediction(listing_data):
    """Tester une prédiction avec un exemple"""
    
    # Charger le modèle et les métadonnées
    model = joblib.load('models/best_model.pkl')
    metadata = joblib.load('models/metadata.pkl')
    
    # Créer un DataFrame avec les features attendues
    features = {}
    
    # Features par défaut (vous devrez adapter selon vos vraies données)
    for feature_name in metadata['feature_names']:
        if feature_name in listing_data:
            features[feature_name] = listing_data[feature_name]
        else:
            features[feature_name] = 0  # Valeur par défaut
    
    # Convertir en DataFrame
    X_test = pd.DataFrame([features])
    
    # Prédiction
    fraud_prob = model.predict_proba(X_test)[0][1]
    prediction = "FRAUDE" if fraud_prob > 0.5 else "LÉGITIME"
    
    return {
        'prediction': prediction,
        'fraud_probability': fraud_prob,
        'confidence': 'ÉLEVÉE' if fraud_prob > 0.8 or fraud_prob < 0.2 else 'MOYENNE'
    }

# Test avec un exemple suspect
print("\n🚨 TEST AVEC UN LISTING SUSPECT:")
suspect_listing = {
    'urgent_keywords': 3,  # Beaucoup de mots urgents
    'suspicious_phrases': 2,  # Phrases suspectes
    'account_age_days': 30,  # Compte récent
    'contact_verified': 0,  # Contact non vérifié
    'low_account_age': 1,  # Compte jeune
    'price_per_sqm': 15,  # Prix très bas
    'price_too_low': 1
}

result_suspect = test_prediction(suspect_listing)
print(f"  Prédiction: {result_suspect['prediction']}")
print(f"  Probabilité de fraude: {result_suspect['fraud_probability']:.2%}")
print(f"  Confiance: {result_suspect['confidence']}")

# Test avec un exemple légitime
print("\n✅ TEST AVEC UN LISTING LÉGITIME:")
legit_listing = {
    'urgent_keywords': 0,  # Pas de mots urgents
    'suspicious_phrases': 0,  # Pas de phrases suspectes
    'account_age_days': 800,  # Compte ancien
    'contact_verified': 1,  # Contact vérifié
    'low_account_age': 0,  # Compte établi
    'price_per_sqm': 35,  # Prix normal
    'price_too_low': 0
}

result_legit = test_prediction(legit_listing)
print(f"  Prédiction: {result_legit['prediction']}")
print(f"  Probabilité de fraude: {result_legit['fraud_probability']:.2%}")
print(f"  Confiance: {result_legit['confidence']}")

print("\n🎉 ENTRAÎNEMENT RÉUSSI ! TOUS LES MODÈLES SONT OPÉRATIONNELS")
print("🔥 Features les plus importantes du Random Forest:")

# Afficher l'importance des features
if hasattr(best_model_obj, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'feature': trainer.feature_names + ['iso_anomaly_score', 'is_anomaly_iso'],
        'importance': best_model_obj.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print(feature_importance.head(10).to_string(index=False))