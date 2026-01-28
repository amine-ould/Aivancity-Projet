#!/usr/bin/env python
"""
Script d'entraînement SIMPLIFIÉ et RAPIDE avec GPU + barre de progression
"""

import os
import sys
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import xgboost as xgb
import lightgbm as lgb
import joblib
from datetime import datetime
from tqdm import tqdm
import logging

# Configuration du logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# Configuration
DATA_PATH = r"data\processed\cleaned_data\sensor_data_cleaned.csv"
MODELS_DIR = "src/models/models"
TARGET_COLUMN = "failure_within_24h"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Créer le répertoire pour les modèles
os.makedirs(MODELS_DIR, exist_ok=True)

print("\n" + "="*70)
print("🚀 ENTRAÎNEMENT RAPIDE AVEC GPU + BARRE DE PROGRESSION")
print("="*70 + "\n")

# ===== 1. CHARGER LES DONNÉES =====
print("📊 Chargement des données...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

print(f"✅ Données chargées: {X.shape[0]} lignes, {X.shape[1]} colonnes")
print(f"✅ Classes: {dict(y.value_counts())}\n")

# ===== 2. SPLIT TRAIN/TEST =====
print("🔀 Split train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

# ===== 3. DÉFINIR LES MODÈLES (GRILLE SIMPLIFIÉE) =====
models_config = {
    'logistic_regression': {
        'model': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'params': {
            'C': [1, 10]
        }
    },
    'random_forest': {
        'model': RandomForestClassifier(n_jobs=1, random_state=RANDOM_STATE),
        'params': {
            'n_estimators': [50],
            'max_depth': [10]
        }
    },
    'gradient_boosting': {
        'model': GradientBoostingClassifier(random_state=RANDOM_STATE),
        'params': {
            'n_estimators': [50],
            'learning_rate': [0.1]
        }
    },
    'xgboost': {
        'model': xgb.XGBClassifier(
            tree_method='hist',  # CPU (plus stable)
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            n_jobs=2
        ),
        'params': {
            'n_estimators': [50],
            'learning_rate': [0.1]
        }
    },
    'lightgbm': {
        'model': lgb.LGBMClassifier(
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=2
        ),
        'params': {
            'n_estimators': [50],
            'learning_rate': [0.1]
        }
    }
}

# ===== 4. ENTRAÎNER LES MODÈLES =====
results = {}
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("🤖 ENTRAÎNEMENT DES MODÈLES\n")
for model_name in tqdm(models_config.keys(), desc="Modèles", unit="model"):
    try:
        print(f"\n⚙️  {model_name.upper()}")
        
        model_info = models_config[model_name]
        
        # GridSearch simplifié
        grid_search = GridSearchCV(
            estimator=model_info['model'],
            param_grid=model_info['params'],
            cv=3,  # 3 folds au lieu de 5
            scoring='roc_auc',
            n_jobs=1,
            verbose=0
        )
        
        # Entraîner
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        # Prédire
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        # Évaluer
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        # Sauvegarder
        model_path = os.path.join(MODELS_DIR, f"{model_name}_{timestamp}.pkl")
        joblib.dump(best_model, model_path)
        
        results[model_name] = {
            'accuracy': accuracy,
            'auc': auc,
            'best_params': grid_search.best_params_,
            'model_path': model_path
        }
        
        print(f"   ✅ Accuracy: {accuracy:.4f}")
        print(f"   ✅ AUC: {auc:.4f}")
        print(f"   ✅ Sauvegardé: {model_path}")
        
    except Exception as e:
        print(f"   ❌ Erreur: {str(e)}")

# ===== 5. RÉSUMÉ FINAL =====
print("\n" + "="*70)
print("📊 RÉSUMÉ DES PERFORMANCES")
print("="*70 + "\n")

for model_name, metrics in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
    print(f"{model_name.upper():20} | Accuracy: {metrics['accuracy']:.4f} | AUC: {metrics['auc']:.4f}")

print("\n" + "="*70)
print("✅ ENTRAÎNEMENT TERMINÉ!")
print("="*70)
print(f"\n📁 Modèles sauvegardés dans: {os.path.abspath(MODELS_DIR)}")
print(f"⏱️  Timestamp: {timestamp}\n")
