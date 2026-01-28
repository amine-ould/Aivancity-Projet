import pandas as pd
import numpy as np
import os
import logging
from datetime import datetime

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("clean_log.log"),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger('clean')

def clean_data(input_dir='extracted_data', output_dir='data/processed/cleaned_data'):
    """
    Nettoie les données extraites et les sauvegarde.
    
    Args:
        input_dir (str): Répertoire contenant les données extraites
        output_dir (str): Répertoire pour les données nettoyées
        
    Returns:
        tuple: (DataFrame capteurs nettoyé, DataFrame défaillances nettoyé)
    """
    try:
        # Création du répertoire de sortie
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            logger.info(f"Répertoire créé: {output_dir}")
        
        # Chargement des données extraites
        sensor_data_path = os.path.join(input_dir, 'sensor_data_extracted.parquet')
        failure_data_path = os.path.join(input_dir, 'failure_data_extracted.parquet')
        
        logger.info(f"Chargement des données capteurs depuis {sensor_data_path}")
        sensor_df = pd.read_parquet(sensor_data_path)
        
        logger.info(f"Chargement des données de défaillance depuis {failure_data_path}")
        failure_df = pd.read_parquet(failure_data_path)
        
        # --- Nettoyage des données capteurs ---
        logger.info("Nettoyage des données capteurs...")
        
        # 1. Remplacer les valeurs infinies et NaN
        sensor_df = sensor_df.replace([np.inf, -np.inf], np.nan)
        
        # 2. Remplir les NaN avec la moyenne par colonne
        numeric_cols = sensor_df.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            sensor_df[col] = sensor_df[col].fillna(sensor_df[col].mean())
        
        # 3. Supprimer les doublons
        original_len = len(sensor_df)
        sensor_df = sensor_df.drop_duplicates()
        logger.info(f"Doublons supprimés: {original_len - len(sensor_df)}")
        
        # 4. Trier par équipement et timestamp
        sensor_df = sensor_df.sort_values(by=['equipment_id', 'timestamp'])
        
        # --- Nettoyage des données de défaillance ---
        logger.info("Nettoyage des données de défaillance...")
        
        # 1. Remplir les valeurs manquantes
        failure_df = failure_df.fillna(failure_df.median(numeric_only=True))
        
        # 2. Supprimer les doublons
        original_len_failure = len(failure_df)
        failure_df = failure_df.drop_duplicates()
        logger.info(f"Doublons supprimés (défaillances): {original_len_failure - len(failure_df)}")
        
        # 3. Garder seulement les défaillances pour les équipements existants
        valid_equipment_ids = sensor_df['equipment_id'].unique()
        failure_df = failure_df[failure_df['equipment_id'].isin(valid_equipment_ids)]
        
        # --- Sauvegarde des données nettoyées ---
        sensor_output = os.path.join(output_dir, 'sensor_data_cleaned.csv')
        failure_output = os.path.join(output_dir, 'failure_data_cleaned.csv')
        
        sensor_df.to_csv(sensor_output, index=False)
        failure_df.to_csv(failure_output, index=False)
        
        logger.info(f"Données capteurs sauvegardées: {sensor_output}")
        logger.info(f"Données défaillances sauvegardées: {failure_output}")
        logger.info(f"Forme finale capteurs: {sensor_df.shape}")
        logger.info(f"Forme finale défaillances: {failure_df.shape}")
        
        print("\n✅ Nettoyage des données réussi!")
        print(f"✅ Fichier capteurs: {sensor_output}")
        print(f"✅ Fichier défaillances: {failure_output}")
        
        return sensor_df, failure_df
        
    except Exception as e:
        logger.error(f"Erreur lors du nettoyage des données: {str(e)}")
        raise

if __name__ == "__main__":
    print("\n" + "="*60)
    print("NETTOYAGE DES DONNÉES")
    print("="*60 + "\n")
    
    clean_sensor_df, clean_failure_df = clean_data()
