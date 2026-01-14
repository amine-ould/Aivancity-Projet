#!/usr/bin/env python
"""
Script simple pour lancer l'entraînement des modèles.
Utilise les données nettoyées d'extract.py et clean.py
"""

import os
import sys
from pathlib import Path

# Ajouter le chemin du dossier src pour les imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))

from models.train_model import train_and_evaluate

# === CONFIGURATION À MODIFIER SELON VOS BESOIN S ===

# 1. ✅ CHEMIN DE DONNÉES - À REMPLACER PAR VOTRE FICHIER PRÉTRAITÉ
# Le fichier doit être un CSV avec une colonne 'failure_within_24h' (la cible)
DATA_PATH = r"data\processed\cleaned_data\sensor_data_cleaned.csv"  # À ADAPTER!

# 2. ✅ RÉPERTOIRE POUR SAUVEGARDER LES MODÈLES
MODELS_DIR = os.path.join("src", "models", "models")

# 3. ✅ COLONNE CIBLE (column with 0s and 1s for failure/no failure)
TARGET_COLUMN = "failure_within_24h"

# 4. ✅ PARAMÈTRES D'ENTRAÎNEMENT
TEST_SIZE = 0.2  # 20% pour test, 80% pour train
RANDOM_STATE = 42  # Pour reproductibilité
CV = 5  # 5-fold cross-validation

# 5. ✅ QUELS MODÈLES ENTRAÎNER? (laissez None pour tous)
# Options: ["random_forest", "gradient_boosting", "logistic_regression", "xgboost", "lightgbm"]
# Mettez None ou [] pour entraîner TOUS les modèles
MODELS_TO_TRAIN = None  # Laissez None pour tous, ou spécifiez: ["random_forest", "xgboost"]

# ============================================================

if __name__ == "__main__":
    print("\n" + "="*60)
    print("DÉMARRAGE DU PIPELINE D'ENTRAÎNEMENT")
    print("="*60 + "\n")
    
    # Vérifier que le fichier de données existe
    if not os.path.exists(DATA_PATH):
        print(f"❌ ERREUR: Le fichier de données n'existe pas: {DATA_PATH}")
        print(f"\nConseil: Vérifiez que:")
        print("  1. Vous avez exécuté extract.py (extrait les données brutes)")
        print("  2. Vous avez exécuté clean.py (nettoie les données)")
        print("  3. Le chemin DATA_PATH est correct")
        print(f"\nChemin attendu: {os.path.abspath(DATA_PATH)}")
        sys.exit(1)
    
    # Créer le répertoire pour les modèles s'il n'existe pas
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print(f"✅ Fichier de données: {os.path.abspath(DATA_PATH)}")
    print(f"✅ Répertoire de sortie: {os.path.abspath(MODELS_DIR)}")
    print(f"✅ Colonne cible: {TARGET_COLUMN}")
    print(f"✅ Modèles à entraîner: {MODELS_TO_TRAIN if MODELS_TO_TRAIN else 'TOUS'}")
    print(f"✅ Train/Test split: {(1-TEST_SIZE)*100:.0f}% / {TEST_SIZE*100:.0f}%")
    print(f"✅ Validation croisée: {CV}-fold\n")
    
    try:
        # Lancer l'entraînement
        trainer, trained_models, evaluation_results, model_paths = train_and_evaluate(
            data_path=DATA_PATH,
            target_column=TARGET_COLUMN,
            models_to_train=MODELS_TO_TRAIN,
            models_dir=MODELS_DIR,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE,
            cv=CV
        )
        
        print("\n" + "="*60)
        print("✅ ENTRAÎNEMENT RÉUSSI!")
        print("="*60)
        print(f"\nModèles entraînés: {list(trained_models.keys())}")
        print(f"Modèles sauvegardés dans: {os.path.abspath(MODELS_DIR)}\n")
        
        # Afficher un résumé des performances
        print("RÉSUMÉ DES PERFORMANCES:")
        print("-" * 50)
        for model_name, eval_info in evaluation_results.items():
            print(f"\n{model_name.upper()}")
            print(f"  Accuracy: {eval_info['accuracy']:.4f}")
            print(f"  AUC:      {eval_info['auc']:.4f}")
        
        print("\n✅ Les fichiers features_importance ont également été sauvegardés.")
        print("   Utilisez-les pour comprendre quelles caractéristiques sont les plus importantes.\n")
        
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
