# Guide de l’application (détails complets : modèle + entraînement)

## 1) Vue d’ensemble
Ce projet construit un **pipeline ML complet** pour prédire les pannes d’équipements à partir de données capteurs. Le flux suit la logique suivante :

1. **Extraction** des données brutes (capteurs + logs de pannes)
2. **Nettoyage** et consolidation
3. **Création de la cible** (panne dans les 24h)
4. **Préparation des features** (sélection, transformations)
5. **Entraînement & évaluation** de plusieurs modèles
6. **Sauvegarde** des meilleurs modèles et artefacts

Le tout est orchestré principalement par `run_training.py`.

---

## 2) Structure du projet

### Dossiers principaux
- `src/`
  - `data/` : extraction et nettoyage
  - `features/` : préparation/ingénierie des features
  - `models/` : entraînement, évaluation, prédiction
- `data/`
  - `raw/` : fichiers bruts
  - `processed/` : données nettoyées et prêtes
- `extracted_data/` : données extraites en formats intermédiaires
- `wandb/` : intégration W&B (tracking)

### Scripts racine
- `run_training.py` : pipeline d’entraînement principal
- `run_training_fast.py` : version optimisée (plus rapide)
- `train_final.py` : entraînement final (paramètres “production”)
- `train_gpu.py` : version GPU si disponible
- `create_target.py` : génération de la cible
- `prepare_features.py` : sélection et préparation des features

---

## 3) Détails du pipeline data

### 3.1 Extraction
**Script :** `src/data/extract.py`

**Objectif :** importer les fichiers bruts et produire des formats cohérents.

**Entrées :**
- `data/raw/predictive_maintenance_sensor_data.csv`
- `data/raw/predictive_maintenance_failure_logs.csv`

**Sorties :** (exemples)
- `extracted_data/sensor_data_extracted.parquet`
- `extracted_data/failure_data_extracted.parquet`

**Ce que fait le script :**
- charge les CSV
- convertit les timestamps en datetime
- assure l’alignement des colonnes
- exporte en Parquet (plus rapide pour la suite)

### 3.2 Nettoyage
**Scripts :** `src/data/clean_simple.py` (rapide) ou `src/data/clean.py` (complet)

**Objectif :** rendre les données exploitables pour le ML.

**Sorties :**
- `data/processed/cleaned_data/sensor_data_cleaned.csv`
- `data/processed/cleaned_data/failure_data_cleaned.csv`

**Actions typiques :**
- suppression/gestion des valeurs manquantes
- suppression des doublons
- contrôle des valeurs aberrantes
- normalisation des types

### 3.3 Création de la cible
**Script :** `create_target.py`

**Objectif :** créer `failure_within_24h` :
- **1** si une panne survient dans les 24h suivant un point capteur
- **0** sinon

**Logique :**
Pour chaque panne connue, marquer les 24h précédentes comme positives.

### 3.4 Préparation des features
**Scripts :** `prepare_features.py` ou `src/features/build_features.py`

**Objectif :** préparer X et y pour l’entraînement.

**Actions typiques :**
- suppression des colonnes non numériques (ex: timestamp, equipment_id)
- sélection des features pertinentes
- ajout de variables dérivées (si `build_features.py`)

---

## 4) Entraînement des modèles

### 4.1 Script principal
**Script :** `run_training.py`

**Rôle :**
- charge les données préparées
- sépare train/test
- entraîne plusieurs modèles
- évalue et loggue les métriques
- sauvegarde les modèles

### 4.2 Paramètres principaux
À adapter dans `run_training.py` :

- `DATA_PATH` : chemin vers le fichier prétraité
- `TARGET_COLUMN` : par défaut `failure_within_24h`
- `TEST_SIZE` : proportion du jeu de test
- `CV` : nombre de folds pour validation croisée
- `MODELS_TO_TRAIN` : liste de modèles à entraîner

### 4.3 Modèles disponibles
- **Random Forest** : robuste, bon sur données bruitées
- **Gradient Boosting** : performant mais plus lent
- **Logistic Regression** : baseline rapide et interprétable
- **XGBoost** : très performant, supporte GPU
- **LightGBM** : rapide et efficace

### 4.4 Évaluation
Les métriques typiques :
- Accuracy
- Precision / Recall / F1
- AUC ROC
- Matrices de confusion

---

## 5) Sorties générées
Après entraînement :

### 5.1 Modèles
Stockés dans `src/models/models/` :
- `random_forest_YYYYMMDD_HHMMSS.pkl`
- `xgboost_YYYYMMDD_HHMMSS.pkl`

### 5.2 Résumés
- fichiers de synthèse des métriques
- importance des features (CSV)
- artefacts WandB si activé

---

## 6) Inférence (prédiction)
**Script :** `src/models/predict_model.py`

**Utilisation :**
- charger un modèle sauvegardé
- fournir un dataset au format compatible
- obtenir une prédiction (0/1 + probabilité)

---

## 7) Exécution complète (pas à pas)

1. Extraction :
```bash
python src/data/extract.py
```

2. Nettoyage :
```bash
python src/data/clean_simple.py
```

3. Cible :
```bash
python create_target.py
```

4. Features :
```bash
python prepare_features.py
```

5. Entraînement :
```bash
python run_training.py
```

---

## 8) Variantes d’entraînement
- **Rapide** : `python run_training_fast.py`
- **Final** : `python train_final.py`
- **GPU** : `python train_gpu.py` (si GPU dispo)

---

## 9) Conseils de production
- figer la version des dépendances (`requirements.txt`)
- sauvegarder les artefacts (modèles + métriques)
- activer W&B pour suivre toutes les expériences
- préparer un pipeline d’inférence dédié

---

Si tu veux une version “one‑command” (orchestration totale) ou un mode API, je peux te proposer une structure dédiée.
