"""
Script pour créer la colonne cible 'failure_within_24h' basée sur les défaillances
"""

import pandas as pd
import os
from datetime import timedelta

# Charger les données nettoyées
sensor_path = "data/processed/cleaned_data/sensor_data_cleaned.csv"
failure_path = "data/processed/cleaned_data/failure_data_cleaned.csv"

print("Chargement des données...")
sensor_df = pd.read_csv(sensor_path)
failure_df = pd.read_csv(failure_path)

# Convertir timestamps
sensor_df['timestamp'] = pd.to_datetime(sensor_df['timestamp'])
failure_df['failure_timestamp'] = pd.to_datetime(failure_df['failure_timestamp'])

print(f"Données capteurs: {sensor_df.shape}")
print(f"Données défaillances: {failure_df.shape}")

# Créer une colonne cible
sensor_df['failure_within_24h'] = 0

# Pour chaque défaillance, marquer les enregistrements 24h avant comme positifs
for idx, failure_row in failure_df.iterrows():
    equipment_id = failure_row['equipment_id']
    failure_time = failure_row['failure_timestamp']
    start_time = failure_time - timedelta(hours=24)
    
    # Marquer les capteurs de cet équipement entre start_time et failure_time
    mask = (sensor_df['equipment_id'] == equipment_id) & \
           (sensor_df['timestamp'] >= start_time) & \
           (sensor_df['timestamp'] < failure_time)
    sensor_df.loc[mask, 'failure_within_24h'] = 1

# Statistiques
print(f"\nColonne cible créée:")
print(f"  - Total d'enregistrements: {len(sensor_df)}")
print(f"  - Positifs (failure_within_24h=1): {(sensor_df['failure_within_24h']==1).sum()}")
print(f"  - Négatifs (failure_within_24h=0): {(sensor_df['failure_within_24h']==0).sum()}")

# Sauvegarder
sensor_df.to_csv(sensor_path, index=False)
print(f"\n✅ Fichier sauvegardé: {sensor_path}")
