"""
Script pour préparer les features pour l'entraînement ML
- Supprime les colonnes non numériques
- Garde seulement les features et la cible
"""

import pandas as pd
import numpy as np

# Charger les données
sensor_path = "data/processed/cleaned_data/sensor_data_cleaned.csv"

print("Chargement des données...")
df = pd.read_csv(sensor_path)

print(f"\nColonnes originales: {df.columns.tolist()}")
print(f"Types: \n{df.dtypes}")

# Garder seulement les colonnes numériques + la cible
# Les features numériques sont: temperature, vibration, pressure, current
# La cible est: failure_within_24h

numeric_features = ['temperature', 'vibration', 'pressure', 'current']
target = 'failure_within_24h'

# Vérifier que les colonnes existent
available_cols = [col for col in numeric_features if col in df.columns]
print(f"\nFeatures numériques disponibles: {available_cols}")
print(f"Cible: {target}")

# Créer le dataframe final avec seulement les features numériques et la cible
if target in df.columns:
    df_final = df[available_cols + [target]].copy()
else:
    df_final = df[available_cols].copy()

print(f"\nShape final: {df_final.shape}")
print(f"Colonnes finales: {df_final.columns.tolist()}")
print(f"\nVérification des NaN:")
print(df_final.isnull().sum())

# Vérifier qu'il n'y a pas de NaN
df_final = df_final.dropna()
print(f"Shape après suppression des NaN: {df_final.shape}")

# Sauvegarder
output_path = "data/processed/cleaned_data/sensor_data_cleaned.csv"
df_final.to_csv(output_path, index=False)
print(f"\n✅ Fichier préparé sauvegardé: {output_path}")
print(f"✅ Prêt pour l'entraînement!")
