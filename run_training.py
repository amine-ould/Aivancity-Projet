#!/usr/bin/env python
"""
Script simple pour lancer l'entraînement des modèles.
Utilise les données nettoyées d'extract.py et clean.py
🎯 Avec suivi WandB pour l'expérience tracking
⚡ Avec barre de progression et optimisation de temps
"""

import os
import sys
from pathlib import Path
import json
from datetime import datetime
import time
import logging
from tqdm import tqdm

# 🔇 Réduire la verbosité des logs pendant l'entraînement
logging.getLogger('models.train_model').setLevel(logging.WARNING)
logging.getLogger('lightgbm').setLevel(logging.WARNING)
logging.getLogger('xgboost').setLevel(logging.WARNING)

# Charger la configuration WandB
sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'wandb'))
from wandb_config import load_wandb_config
load_wandb_config()

import wandb
from wandb_metrics_logger import WandBMetricsLogger

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
CV = 3  # ⚡ Réduit de 5 à 3 pour la vitesse (minimal pour la validité statistique)

# 5. ✅ QUELS MODÈLES ENTRAÎNER? (laissez None pour tous)
# Options: ["random_forest", "gradient_boosting", "logistic_regression", "xgboost", "lightgbm"]
# Mettez None ou [] pour entraîner TOUS les modèles
# ⚡ OPTIMISÉ: Entraîner les modèles rapides par défaut
MODELS_TO_TRAIN = ["random_forest", "gradient_boosting", "logistic_regression", "xgboost", "lightgbm"]  # ⚡ Rapides seulement (3-5 min)

# 6. ✅ CONFIGURATION WANDB
WANDB_CONFIG = {
    "project": "industrial-failure-prediction",  # Changez le nom du projet
    "entity": "ouldamroucheamine-aivancity-school-for-technology-busine",  # Workspace WandB
    "enable_wandb": True,  # Mettez False pour désactiver WandB temporairement
    "tags": ["training", "production"],
    "notes": "Entraînement complet avec tous les modèles"
}

# ============================================================

if __name__ == "__main__":
    # ⏱️ Démarrer le chrono
    start_time = time.time()
    
    print("\n" + "="*60)
    print("⚡ PIPELINE D'ENTRAÎNEMENT OPTIMISÉ")
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
    print(f"⚡ Validation croisée: {CV}-fold (optimisé)")
    print(f"⚡ Modèles rapides uniquement")
    print(f"✅ Modèles à entraîner: {MODELS_TO_TRAIN if MODELS_TO_TRAIN else 'TOUS'}")
    print(f"✅ Train/Test split: {(1-TEST_SIZE)*100:.0f}% / {TEST_SIZE*100:.0f}%")
    print(f"✅ Validation croisée: {CV}-fold\n")
    
    # === INITIALISER WANDB ===
    wandb_run = None
    data_for_logging = None
    if WANDB_CONFIG.get("enable_wandb", True):
        try:
            wandb_run = wandb.init(
                project=WANDB_CONFIG["project"],
                entity=WANDB_CONFIG.get("entity"),
                tags=WANDB_CONFIG.get("tags", []),
                notes=WANDB_CONFIG.get("notes", ""),
                config={
                    "data_path": DATA_PATH,
                    "target_column": TARGET_COLUMN,
                    "test_size": TEST_SIZE,
                    "random_state": RANDOM_STATE,
                    "cv_folds": CV,
                    "models_to_train": MODELS_TO_TRAIN if MODELS_TO_TRAIN else "ALL"
                }
            )
            print(f"🎯 WandB initialisé: {wandb_run.get_url()}\n")

            # ===== Logs dataset & contexte (pertinent avant entraînement) =====
            try:
                import pandas as pd
                import numpy as np
                import matplotlib.pyplot as plt
                import seaborn as sns
                data_for_logging = pd.read_csv(DATA_PATH)

                # Config enrichie
                wandb.config.update({
                    "data_rows": int(data_for_logging.shape[0]),
                    "data_cols": int(data_for_logging.shape[1]),
                    "features_count": int(data_for_logging.shape[1] - 1),
                }, allow_val_change=True)

                # Artifact dataset
                dataset_artifact = wandb.Artifact(
                    name="predictive_maintenance_dataset",
                    type="dataset",
                    description="Dataset prétraité utilisé pour l'entraînement",
                    metadata={
                        "rows": int(data_for_logging.shape[0]),
                        "cols": int(data_for_logging.shape[1]),
                        "target_column": TARGET_COLUMN
                    }
                )
                dataset_artifact.add_file(DATA_PATH)
                wandb_run.log_artifact(dataset_artifact)

                # Échantillon du dataset
                wandb.log({"data/sample": wandb.Table(dataframe=data_for_logging.head(200))})

                # Taux de valeurs manquantes (Top 30)
                missing_rate = data_for_logging.isnull().mean().sort_values(ascending=False)
                missing_df = missing_rate.head(30).reset_index()
                missing_df.columns = ["feature", "missing_rate"]
                wandb.log({"data/missing_rate": wandb.Table(dataframe=missing_df)})

                # Statistiques descriptives
                desc_df = data_for_logging.describe(include="all").transpose().reset_index()
                desc_df.columns = ["feature"] + [c if c else "value" for c in desc_df.columns[1:]]
                wandb.log({"data/describe": wandb.Table(dataframe=desc_df)})

                # Heatmap corrélation (features numériques)
                numeric_df = data_for_logging.select_dtypes(include=[np.number])
                if TARGET_COLUMN in numeric_df.columns:
                    numeric_df = numeric_df.drop(columns=[TARGET_COLUMN])
                if numeric_df.shape[1] > 1:
                    corr = numeric_df.corr()
                    fig, ax = plt.subplots(figsize=(10, 8))
                    sns.heatmap(corr, cmap="coolwarm", center=0, ax=ax)
                    ax.set_title("Correlation Heatmap (features numériques)")
                    wandb.log({"data/correlation_heatmap": wandb.Image(fig)})
                    plt.close(fig)

                # Distribution cible
                if TARGET_COLUMN in data_for_logging.columns:
                    target_counts = data_for_logging[TARGET_COLUMN].value_counts()
                    target_ratio = data_for_logging[TARGET_COLUMN].value_counts(normalize=True)
                    wandb.log({
                        "data/target_pos": int(target_counts.get(1, 0)),
                        "data/target_neg": int(target_counts.get(0, 0)),
                        "data/target_pos_rate": float(target_ratio.get(1, 0)),
                        "data/target_neg_rate": float(target_ratio.get(0, 0))
                    })

                    # Bar chart distribution cible
                    fig, ax = plt.subplots(figsize=(6, 4))
                    ax.bar(["neg", "pos"], [int(target_counts.get(0, 0)), int(target_counts.get(1, 0))], color=["#4C78A8", "#F58518"])
                    ax.set_title("Distribution de la cible")
                    ax.set_ylabel("Count")
                    wandb.log({"data/target_distribution": wandb.Image(fig)})
                    plt.close(fig)
            except Exception as e:
                print(f"⚠️ Impossible de logger les infos dataset dans WandB: {e}")
        except Exception as e:
            print(f"⚠️ WandB non disponible: {e}")
            print("  L'entraînement continue sans WandB...\n")
    
    try:
        # 📊 Étape 1: Lancer l'entraînement (avec barre de progression temps réel)
        import threading
        from queue import Queue
        
        training_state = {'status': 'en cours', 'start_time': time.time()}
        
        def update_progress_bar():
            """Barre de progression basée sur le temps + modèles complétés"""
            # Estimation: ~6 min pour 3 modèles (XGBoost 2-3min, LightGBM 2-3min, LogReg 30s)
            estimated_time = 360  # 6 minutes en secondes
            
            with tqdm(total=100, desc="⚙️ Entraînement", 
                     bar_format='{l_bar}{bar}| {n_fmt}% | {elapsed}s') as pbar:
                last_percentage = 0
                
                while training_state['status'] == 'en cours':
                    elapsed = time.time() - training_state['start_time']
                    
                    # Calcul du pourcentage: 0-95% pendant l'entraînement, 95-100% à la fin
                    percentage = min(95, int((elapsed / estimated_time) * 95))
                    
                    if percentage > last_percentage:
                        pbar.update(percentage - last_percentage)
                        last_percentage = percentage
                    
                    time.sleep(1)  # Mise à jour chaque seconde
                
                # Compléter à 100%
                if last_percentage < 100:
                    pbar.update(100 - last_percentage)
        
        # Lancer le thread de progression
        progress_thread = threading.Thread(target=update_progress_bar, daemon=True)
        progress_thread.start()
        
        # Lancer l'entraînement
        trained_models, evaluation_results, model_paths, best_model = train_and_evaluate(
            data_path=DATA_PATH,
            target_column=TARGET_COLUMN,
            models_to_train=MODELS_TO_TRAIN,
            models_dir=MODELS_DIR,
            test_size=TEST_SIZE,
            cv=CV,
            use_wandb=True,
            wandb_run=wandb_run
        )
        
        # Signal que l'entraînement est fini
        training_state['status'] = 'terminé'
        progress_thread.join(timeout=2)
        
        # ⏱️ Calculer le temps total
        elapsed_time = time.time() - start_time
        
        print("\n" + "="*60)
        print("✅ ENTRAÎNEMENT RÉUSSI!")
        print("="*60)
        print(f"\n⏱️  Temps total: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)\n")
        
        print(f"✅ Modèles entraînés: {list(trained_models.keys())}")
        print(f"✅ Sauvegardés dans: {os.path.abspath(MODELS_DIR)}\n")
        
        # 📊 Étape 2: Logger les métriques détaillées dans WandB
        print("📊 Logging des métriques détaillées...\n")
        
        if wandb_run:
            # Charger les données pour feature importance et drift
            if data_for_logging is None:
                import pandas as pd
                data = pd.read_csv(DATA_PATH)
            else:
                data = data_for_logging
            from sklearn.model_selection import train_test_split
            X = data.drop(columns=[TARGET_COLUMN])
            y = data[TARGET_COLUMN]
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            
            # Logger les métriques pour chaque modèle
            for model_name, model_info in tqdm(trained_models.items(), desc="Logging modèles",
                                               bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
                model = model_info['model']
                y_pred = model.predict(X_test)
                y_pred_proba = model.predict_proba(X_test)[:, 1]
                
                # Métriques détaillées
                WandBMetricsLogger.log_model_metrics(model_name, y_test, y_pred, y_pred_proba, model)
                
                # Feature importance
                WandBMetricsLogger.log_feature_importance(
                    model_name, model, list(X_train.columns), top_n=15
                )
                
                # Hyperparamètres
                if 'params' in model_info:
                    WandBMetricsLogger.log_hyperparameters(model_name, model_info['params'])
            
            # Data drift
            try:
                WandBMetricsLogger.log_data_drift(list(X_train.columns), X_train, X_test)
            except Exception as e:
                print(f"⚠️ Impossible de logger drift: {e}")
            
            # Comparaison des modèles
            WandBMetricsLogger.log_model_comparison(evaluation_results)

            # Tableau de synthèse des métriques
            try:
                import pandas as pd
                summary_rows = []
                for model_name, eval_info in evaluation_results.items():
                    summary_rows.append({
                        "model": model_name,
                        "accuracy": float(eval_info.get("accuracy", 0)),
                        "auc": float(eval_info.get("auc", 0)),
                        "recall": float(eval_info.get("recall", 0)),
                        "precision": float(eval_info.get("precision", 0)),
                        "f1": float(eval_info.get("f1", 0))
                    })
                summary_df = pd.DataFrame(summary_rows)
                wandb.log({"results/summary_table": wandb.Table(dataframe=summary_df)})
            except Exception as e:
                print(f"⚠️ Impossible de logger le tableau de synthèse: {e}")
            
            print("✅ Métriques enregistrées dans WandB\n")
        
        # 📊 Étape 3: Afficher le résumé
        print("RÉSUMÉ DES PERFORMANCES:")
        print("-" * 50)
        
        for model_name, eval_info in tqdm(evaluation_results.items(), desc="Résultats", 
                                         bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt}'):
            print(f"\n{model_name.upper()}")
            print(f"  ✅ Accuracy:  {eval_info['accuracy']:.4f}")
            print(f"  ✅ AUC ROC:   {eval_info['auc']:.4f}")
        
        # 📤 Étape 4: Finalize WandB
        if wandb_run:
            print("\n✅ Résultats enregistrés dans WandB")
            try:
                wandb.config.update({
                    "best_model": best_model,
                    "elapsed_seconds": float(elapsed_time)
                }, allow_val_change=True)
                wandb.log({"training/elapsed_seconds": float(elapsed_time)})
            except Exception as e:
                print(f"⚠️ Impossible de logger le temps d'entraînement: {e}")
            wandb.finish()
        
        print("\n✅ Les fichiers features_importance ont également été sauvegardés.")
        print("   Utilisez-les pour comprendre quelles caractéristiques sont les plus importantes.")
        print(f"\n⏱️  Temps total: {elapsed_time:.2f}s ({elapsed_time/60:.2f} min)\n")
        
    except Exception as e:
        print(f"\n❌ ERREUR lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        if wandb_run:
            wandb.finish(exit_code=1)
        sys.exit(1)
