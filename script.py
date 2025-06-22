# Analysons d'abord le dataset pour comprendre sa structure
import pandas as pd
import numpy as np

# Charger le dataset
df = pd.read_csv('singapore_real_estate_fraud_dataset_final.csv')

print("=== ANALYSE DU DATASET ===")
print(f"Shape: {df.shape}")
print(f"\nColonnes: {df.columns.tolist()}")
print(f"\nTypes de données:")
print(df.dtypes)
print(f"\nDistribution des fraudes:")
print(df['is_scam'].value_counts())
print(f"\nPremières lignes:")
print(df.head())
print(f"\nValeurs manquantes:")
print(df.isnull().sum())