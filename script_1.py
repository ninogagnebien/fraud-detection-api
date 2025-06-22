# Créer le dossier models s'il n'existe pas
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import lightgbm as lgb
import pandas as pd
import numpy as np
import re
from datetime import datetime

# Créer le dossier models
os.makedirs('models', exist_ok=True)

# Charger les données
df = pd.read_csv('singapore_real_estate_fraud_dataset_final.csv')

print("🚀 DÉMARRAGE DE L'ENTRAÎNEMENT DES MODÈLES DE DÉTECTION DE FRAUDE")
print("=" * 60)

class FraudDetectionTrainer:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.models = {}
        self.feature_names = []
        
    def extract_features(self, df):
        """Extraction des features pour la détection de fraude"""
        print("📊 Extraction des features...")
        
        features_df = pd.DataFrame()
        
        # 1. Features textuelles basiques
        print("   - Features textuelles...")
        urgent_keywords = ['urgent', 'limited', 'act fast', 'deal ends', 'quick', 'cheap', 'best price']
        suspicious_phrases = ['no viewing', 'pay deposit', 'direct owner', 'no agent', 'fake listings beware']
        
        features_df['urgent_keywords'] = df['title'].apply(
            lambda x: sum(1 for keyword in urgent_keywords if keyword.lower() in str(x).lower())
        )
        
        features_df['suspicious_phrases'] = df['description'].apply(
            lambda x: sum(1 for phrase in suspicious_phrases if phrase.lower() in str(x).lower())
        )
        
        features_df['title_length'] = df['title'].apply(len)
        features_df['description_length'] = df['description'].apply(len)
        features_df['title_exclamation'] = df['title'].apply(lambda x: str(x).count('!'))
        
        # 2. Features numériques
        print("   - Features numériques...")
        features_df['price'] = df['price']
        features_df['area_sqm'] = df['area_sqm']
        features_df['price_per_sqm'] = df['price_per_sqm']
        features_df['account_age_days'] = df['account_age_days']
        
        # Features dérivées des prix
        location_stats = df.groupby('location')['price_per_sqm'].agg(['mean', 'std']).reset_index()
        df_with_stats = df.merge(location_stats, on='location', how='left')
        features_df['price_deviation'] = abs(df_with_stats['price_per_sqm'] - df_with_stats['mean']) / (df_with_stats['std'] + 1e-8)
        
        # 3. Features de profil utilisateur
        print("   - Features utilisateur...")
        features_df['contact_verified'] = df['contact_verified'].astype(int)
        features_df['low_account_age'] = (df['account_age_days'] < 152).astype(int)
        
        # Statistics par utilisateur
        user_stats = df.groupby('user_id').agg({
            'listing_id': 'count',
            'price': ['mean', 'std'],
            'location': 'nunique'
        }).reset_index()
        
        user_stats.columns = ['user_id', 'user_listing_count', 'user_avg_price', 'user_price_std', 'user_location_diversity']
        user_stats['user_price_std'] = user_stats['user_price_std'].fillna(0)
        
        df_with_user = df.merge(user_stats, on='user_id', how='left')
        features_df['user_listing_count'] = df_with_user['user_listing_count']
        features_df['user_avg_price'] = df_with_user['user_avg_price']
        features_df['user_price_variability'] = df_with_user['user_price_std']
        features_df['user_location_diversity'] = df_with_user['user_location_diversity']
        
        # 4. Encodage de la localisation
        print("   - Encodage localisation...")
        le_location = LabelEncoder()
        features_df['location_encoded'] = le_location.fit_transform(df['location'])
        self.label_encoders['location'] = le_location
        
        # 5. Features d'anomalies de prix
        features_df['price_too_low'] = (df['price_per_sqm'] < 20).astype(int)
        features_df['price_very_high'] = (df['price_per_sqm'] > 55).astype(int)
        
        # Gérer les valeurs manquantes
        features_df = features_df.fillna(0)
        
        self.feature_names = features_df.columns.tolist()
        print(f"   ✅ {len(self.feature_names)} features extraites")
        
        return features_df
    
    def train_anomaly_models(self, X, y):
        """Entraînement des modèles de détection d'anomalies"""
        print("🔍 Entraînement des modèles de détection d'anomalies...")
        
        # Normaliser les données pour les modèles d'anomalies
        X_scaled = self.scaler.fit_transform(X)
        
        # Isolation Forest
        print("   - Isolation Forest...")
        iso_forest = IsolationForest(
            contamination=0.2,  # Estimation basée sur notre ratio 200/1000
            random_state=42,
            n_estimators=100
        )
        iso_scores = iso_forest.fit_predict(X_scaled)
        iso_scores_prob = iso_forest.decision_function(X_scaled)
        
        # Sauvegarder le modèle
        joblib.dump(iso_forest, 'models/isolation_forest.pkl')
        joblib.dump(self.scaler, 'models/scaler.pkl')
        
        # Évaluer la performance
        iso_binary = (iso_scores == -1).astype(int)
        iso_auc = roc_auc_score(y, -iso_scores_prob)  # Négatif car plus négatif = plus anormal
        
        print(f"   ✅ Isolation Forest AUC: {iso_auc:.4f}")
        
        self.models['isolation_forest'] = iso_forest
        
        return {
            'iso_anomaly_score': iso_scores_prob,
            'is_anomaly_iso': iso_binary
        }
    
    def train_supervised_models(self, X, y):
        """Entraînement des modèles supervisés"""
        print("🤖 Entraînement des modèles supervisés...")
        
        # Division des données
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.3, random_state=42, stratify=y
        )
        
        results = {}
        
        # 1. Random Forest
        print("   - Random Forest...")
        rf_model = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        rf_model.fit(X_train, y_train)
        
        y_pred_rf = rf_model.predict(X_test)
        y_pred_proba_rf = rf_model.predict_proba(X_test)[:, 1]
        rf_auc = roc_auc_score(y_test, y_pred_proba_rf)
        
        # Sauvegarder le modèle
        joblib.dump(rf_model, 'models/random_forest.pkl')
        
        self.models['random_forest'] = rf_model
        results['random_forest'] = {
            'auc': rf_auc,
            'predictions': y_pred_proba_rf,
            'test_labels': y_test
        }
        
        print(f"   ✅ Random Forest AUC: {rf_auc:.4f}")
        
        # 2. Logistic Regression
        print("   - Logistic Regression...")
        lr_model = LogisticRegression(
            random_state=42,
            class_weight='balanced',
            max_iter=1000
        )
        lr_model.fit(X_train, y_train)
        
        y_pred_lr = lr_model.predict(X_test)
        y_pred_proba_lr = lr_model.predict_proba(X_test)[:, 1]
        lr_auc = roc_auc_score(y_test, y_pred_proba_lr)
        
        # Sauvegarder le modèle
        joblib.dump(lr_model, 'models/logistic_regression.pkl')
        
        self.models['logistic_regression'] = lr_model
        results['logistic_regression'] = {
            'auc': lr_auc,
            'predictions': y_pred_proba_lr,
            'test_labels': y_test
        }
        
        print(f"   ✅ Logistic Regression AUC: {lr_auc:.4f}")
        
        # 3. LightGBM
        print("   - LightGBM...")
        lgb_model = lgb.LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=7,
            random_state=42,
            class_weight='balanced',
            verbosity=-1
        )
        lgb_model.fit(X_train, y_train)
        
        y_pred_lgb = lgb_model.predict(X_test)
        y_pred_proba_lgb = lgb_model.predict_proba(X_test)[:, 1]
        lgb_auc = roc_auc_score(y_test, y_pred_proba_lgb)
        
        # Sauvegarder le modèle
        joblib.dump(lgb_model, 'models/lightgbm.pkl')
        
        self.models['lightgbm'] = lgb_model
        results['lightgbm'] = {
            'auc': lgb_auc,
            'predictions': y_pred_proba_lgb,
            'test_labels': y_test
        }
        
        print(f"   ✅ LightGBM AUC: {lgb_auc:.4f}")
        
        return results
    
    def save_metadata(self):
        """Sauvegarder les métadonnées du modèle"""
        metadata = {
            'feature_names': self.feature_names,
            'label_encoders': {k: v.classes_ for k, v in self.label_encoders.items()},
            'training_date': datetime.now().isoformat(),
            'model_version': '1.0',
            'dataset_size': len(df)
        }
        
        joblib.dump(metadata, 'models/metadata.pkl')
        print("   ✅ Métadonnées sauvegardées")
    
    def generate_model_report(self, results):
        """Générer un rapport détaillé des modèles"""
        print("\n📈 RAPPORT DE PERFORMANCE DES MODÈLES")
        print("=" * 50)
        
        best_model = None
        best_auc = 0
        
        for model_name, result in results.items():
            auc = result['auc']
            print(f"\n{model_name.upper()}:")
            print(f"  AUC Score: {auc:.4f}")
            
            # Classification report
            y_pred_binary = (result['predictions'] > 0.5).astype(int)
            print("  Classification Report:")
            print(classification_report(result['test_labels'], y_pred_binary, indent='    '))
            
            if auc > best_auc:
                best_auc = auc
                best_model = model_name
        
        print(f"\n🏆 MEILLEUR MODÈLE: {best_model} (AUC: {best_auc:.4f})")
        
        # Sauvegarder le meilleur modèle avec un nom spécial
        if best_model:
            best_model_obj = self.models[best_model]
            joblib.dump(best_model_obj, 'models/best_model.pkl')
            print(f"   ✅ Meilleur modèle sauvegardé comme 'best_model.pkl'")
            
            # Sauvegarder aussi les informations du meilleur modèle
            best_model_info = {
                'model_name': best_model,
                'auc_score': best_auc,
                'model_path': 'models/best_model.pkl'
            }
            joblib.dump(best_model_info, 'models/best_model_info.pkl')

# Initialiser le trainer
trainer = FraudDetectionTrainer()

# Extraire les features
X = trainer.extract_features(df)
y = df['is_scam']

print(f"✅ Dataset préparé: {X.shape[0]} échantillons, {X.shape[1]} features")