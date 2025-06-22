# Continuer avec l'entraînement des modèles

# Entraîner les modèles de détection d'anomalies
anomaly_features = trainer.train_anomaly_models(X, y)

# Ajouter les features d'anomalies au dataset principal
X_with_anomalies = pd.concat([X, pd.DataFrame(anomaly_features)], axis=1)

print(f"✅ Features avec anomalies: {X_with_anomalies.shape[1]} features")

# Entraîner les modèles supervisés
supervised_results = trainer.train_supervised_models(X_with_anomalies, y)

# Sauvegarder les métadonnées
trainer.save_metadata()

# Générer le rapport
trainer.generate_model_report(supervised_results)