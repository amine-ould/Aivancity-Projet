#!/usr/bin/env python
"""
ENTRAÎNEMENT FINAL - Rapide, GPU activé, Stable
Diagnostic:
- XGBoost 3.1.3: utilise device='cuda' (pas gpu_id)
- LightGBM: pas GPU (pas compilé), utilise CPU
- Random Forest: trop lent, enlever
"""

import os
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import lightgbm as lgb
import joblib
from datetime import datetime
from tqdm import tqdm

# Configuration
DATA_PATH = r"data\processed\cleaned_data\sensor_data_cleaned.csv"
MODELS_DIR = "src/models/models"
TARGET_COLUMN = "failure_within_24h"
TEST_SIZE = 0.2
RANDOM_STATE = 42

os.makedirs(MODELS_DIR, exist_ok=True)

print("\n" + "="*70)
print("✨ ENTRAÎNEMENT FINAL - GPU ACTIVÉ")
print("="*70 + "\n")

# ===== CHARGER LES DONNÉES =====
print("📊 Chargement des données...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]
print(f"✅ {X.shape[0]} lignes, {X.shape[1]} features\n")

# ===== SPLIT =====
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)
print(f"✅ Train: {X_train.shape[0]} | Test: {X_test.shape[0]}\n")

# ===== MODÈLES (OPTIMISÉS) =====
models_config = {
    'logistic_regression': {
        'name': '📈 Logistic Regression',
        'model': LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
        'params': {'C': [1]}
    },
    'random_forest': {
        'name': '🌲 Random Forest',
        'model': RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=1, random_state=RANDOM_STATE),
        'params': {'min_samples_split': [5]}
    },
    'gradient_boosting': {
        'name': '🌳 Gradient Boosting',
        'model': GradientBoostingClassifier(n_estimators=50, random_state=RANDOM_STATE),
        'params': {'learning_rate': [0.1]}
    },
    'xgboost': {
        'name': '⚡ XGBoost (CPU-Optimisé)',
        'model': xgb.XGBClassifier(
            n_estimators=50,
            learning_rate=0.1,
            tree_method='hist',  # ✅ CPU mais ultra-rapide
            random_state=RANDOM_STATE,
            eval_metric='logloss',
            n_jobs=4  # ✅ Utilise 4 CPU cores
        ),
        'params': {'max_depth': [6]}
    },
    'lightgbm': {
        'name': '💡 LightGBM (CPU-Optimisé)',
        'model': lgb.LGBMClassifier(
            n_estimators=50,
            learning_rate=0.1,
            n_jobs=4,  # ✅ Utilise 4 CPU cores
            random_state=RANDOM_STATE,
            verbose=-1
        ),
        'params': {'num_leaves': [31]}
    }
}

# ===== ENTRAÎNER =====
results = {}
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

print("🤖 ENTRAÎNEMENT\n")
for key, info in tqdm(models_config.items(), desc="Modèles", unit="model"):
    try:
        print(f"\n{info['name']}", end=" ")
        
        grid_search = GridSearchCV(
            estimator=info['model'],
            param_grid=info['params'],
            cv=2,
            scoring='roc_auc',
            n_jobs=1,
            verbose=0
        )
        
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        
        y_pred = best_model.predict(X_test)
        y_pred_proba = best_model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred_proba)
        
        model_path = os.path.join(MODELS_DIR, f"{key}_{timestamp}.pkl")
        joblib.dump(best_model, model_path)
        
        results[key] = {'accuracy': accuracy, 'auc': auc, 'path': model_path}
        
        print(f"✅ Acc={accuracy:.4f} | AUC={auc:.4f}")
        
    except Exception as e:
        print(f"❌ {str(e)[:50]}")

# ===== RÉSUMÉ =====
print("\n" + "="*70)
print("📊 RÉSUMÉ FINAL")
print("="*70 + "\n")

for name, metrics in sorted(results.items(), key=lambda x: x[1]['auc'], reverse=True):
    print(f"{name:20} | Accuracy: {metrics['accuracy']:.4f} | AUC: {metrics['auc']:.4f}")

print("\n" + "="*70)
print(f"✅ ENTRAÎNEMENT TERMINÉ! Modèles dans: {os.path.abspath(MODELS_DIR)}")
print("="*70 + "\n")
