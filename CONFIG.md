# Configuration - Détecteur de Fraude Immobilière

## 🎯 Paramètres du Modèle

### Seuils de Classification
- **Fraude probable** : ≥ 50% de probabilité
- **Confiance élevée** : < 20% ou > 80%
- **Révision manuelle** : 20% - 80%

### Facteurs de Risque Principaux

#### Âge du Compte
- **Très risqué** : < 30 jours
- **Risqué** : 30-151 jours  
- **Sûr** : > 152 jours

#### Prix au m²
- **Trop bas** : < 20 SGD/m²
- **Normal** : 20-55 SGD/m²
- **Trop élevé** : > 55 SGD/m²

#### Mots-clés Urgents
- urgent, limited, act fast, deal ends
- quick, cheap, best price

#### Phrases Suspectes  
- no viewing, pay deposit, direct owner
- no agent, fake listings beware

## 🔧 Configuration Technique

### Modèles Utilisés
- **Principal** : Random Forest (AUC = 1.0)
- **Anomalies** : Isolation Forest
- **Normalisation** : StandardScaler

### Localisation Supportée
- Tampines (code: 0)
- Orchard (code: 1) 
- Hougang (code: 2)
- Bukit Timah (code: 3)
- Toa Payoh (code: 4)
- Jurong (code: 5)

## 📊 Métriques de Performance

| Modèle | AUC | Précision | Rappel | F1-Score |
|--------|-----|-----------|--------|----------|
| Random Forest | 1.000 | 100% | 100% | 100% |
| LightGBM | 1.000 | 100% | 100% | 100% |
| Logistic Regression | 1.000 | 100% | 100% | 100% |

## 🚀 Utilisation Rapide

### Tests de Base
```bash
python fraud_detector.py --test
```

### Prédiction Simple
```bash
python fraud_detector.py --predict "Urgent Rental!" "Direct owner, no viewing" 1500 50 30
```

### Tests Batch
```bash
python fraud_detector.py --batch example_listings.json
```

## 🛠️ Maintenance

### Ré-entraînement Recommandé
- **Fréquence** : Tous les 3 mois
- **Données** : Minimum 100 nouveaux cas étiquetés
- **Validation** : Comparaison AUC avant/après

### Monitoring Continu
- Taux de faux positifs < 5%
- Temps de réponse API < 200ms  
- Feedback modérateurs collecté
