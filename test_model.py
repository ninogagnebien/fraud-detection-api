import joblib
import pandas as pd
import numpy as np

def test_trained_model():
    """Tester le modèle entraîné"""
    print("🧪 Test du modèle entraîné...")
    
    try:
        # Charger le modèle et métadonnées
        model = joblib.load('models/fraud_detection_model.pkl')
        metadata = joblib.load('models/metadata.pkl')
        
        print(f"✅ Modèle chargé: {metadata['best_model']}")
        print(f"📊 Features attendues: {metadata['num_features']}")
        
        # Créer des données de test
        test_data = np.random.rand(1, 21)  # 21 features
        
        # Test de prédiction
        prediction = model.predict(test_data)
        probability = model.predict_proba(test_data)[0][1]
        
        print(f"🎯 Test de prédiction réussi!")
        print(f"   Prédiction: {'FRAUDE' if prediction[0] else 'LÉGITIME'}")
        print(f"   Probabilité: {probability:.3f}")
        
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False

if __name__ == "__main__":
    test_trained_model()
