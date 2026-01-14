# 🤖 GUIDE D'ENTRAÎNEMENT DU MODÈLE

## 📋 Résumé du Code

Ton projet contient un **pipeline ML complet** pour prédire les défaillances d'équipements:

```
DONNÉES BRUTES
     ↓
[1] EXTRACTION (extract.py) → Charge les CSV des capteurs et défaillances
     ↓
[2] NETTOYAGE (clean.py) → Supprime les doublons, gère les NaN, outliers
     ↓
[3] FEATURES (build_features.py) → Crée variables polynomiales, cycles
     ↓
[4] ENTRAÎNEMENT ⭐ (train_model.py) → Entraîne 5 modèles différents
     ↓
[5] MODÈLES SAUVEGARDÉS → À utiliser pour prédictions futures
```

---

## 🔴 Les 3 Problèmes qu'il y Avait

### ❌ **Problème #1: Modèles CPU manquants (ligne 96)**
```python
# AVANT (incomplète):
self.models = { }

# APRÈS (corrigée) ✅:
self.models = {
    'random_forest': {...},
    'gradient_boosting': {...},
    'logistic_regression': {...},
    'xgboost': {...},
    'lightgbm': {...}
}
```

### ❌ **Problème #2: Métriques d'évaluation incomplètes (ligne 238)**
```python
# AVANT (incomplet):
accuracy = 
conf_matrix = 
class_report = 
auc_score = 

# APRÈS (corrigée) ✅:
accuracy = (y_pred == y_test).astype(int).mean()
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)
auc_score = roc_auc_score(y_test, y_pred_proba)
```

### ❌ **Problème #3: Fonction `find_best_model()` incomplète**
```python
# AVANT (incomplet):
scores =   # ← vide!
best_model = 

# APRÈS (corrigée) ✅:
scores = {model: eval_info[metric] for model, eval_info in evaluation_results.items()}
best_model = max(scores, key=scores.get)
```

**Tous les problèmes ont été corrigés! ✅**

---

## 🚀 Comment Lancer l'Entraînement

### Étape 1: Préparer les données
Avant l'entraînement, tu dois avoir des données nettoyées. Exécute dans cet ordre:

```bash
# 1. Extraire les données brutes
python src/data/extract.py

# 2. Nettoyer les données
python src/data/clean.py

# 3. Créer les features (optionnel si déjà fait)
python src/features/build_features.py
```

**⚠️ IMPORTANT:** À la fin, tu dois avoir un fichier CSV avec:
- Colonnes de features (temperature, vibration, pressure, current, etc.)
- Une colonne `failure_within_24h` (0 ou 1) = LA CIBLE

### Étape 2: Éditer `run_training.py`

Ouvre le fichier `run_training.py` et change ces lignes:

```python
# ✅ À MODIFIER - Chemin vers votre fichier prétraité
DATA_PATH = r"data\processed\cleaned_data\VOTRE_FICHIER.csv"

# ✅ Les autres paramètres (facultatif)
TEST_SIZE = 0.2  # 80% train, 20% test
TARGET_COLUMN = "failure_within_24h"  # ← Colonne de prédiction
MODELS_TO_TRAIN = None  # Laissez None pour tous, ou ["random_forest", "xgboost"]
```

### Étape 3: Lancer l'entraînement

```bash
python run_training.py
```

**Ou avec la ligne de commande (avancé):**

```bash
python -m src.models.train_model --data_path "chemin/donnees.csv" \
                                  --target_column "failure_within_24h" \
                                  --models random_forest xgboost
```

---

## 📊 Quels Modèles Sont Entraînés?

Le code entraîne **5 modèles différents**:

| Modèle | Description | Temps | Performance |
|--------|-------------|-------|-------------|
| **Random Forest** 🌲🌲 | Ensemble de 300 arbres | Moyen | Très bon |
| **Gradient Boosting** 📈 | Boosting des arbres | Lent | Excellent |
| **Logistic Regression** 📊 | Régression linéaire | Rapide | Bon |
| **XGBoost** ⚡ | Boosting GPU-ready | Rapide | Excellent |
| **LightGBM** 💡 | Boosting léger | Super rapide | Excellent |

---

## 📁 Où Sont Sauvegardés les Modèles?

Après l'entraînement, tu trouveras:

```
src/models/models/
├── random_forest_20250114_143022.pkl
├── xgboost_20250114_143022.pkl
├── random_forest_feature_importance_20250114_143022.csv
├── xgboost_feature_importance_20250114_143022.csv
└── training_summary_20250114_143022.pkl
```

**Fichiers créés:**
- `*.pkl` = Le modèle complet (poids + paramètres)
- `*_feature_importance.csv` = Quelles features sont les plus importantes
- `training_summary_*.pkl` = Résumé de tous les résultats

---

## 🔍 Que Fait Exactement le Code?

### 1️⃣ **Load Data** (`load_data()`)
```python
Charge le fichier CSV prétraité dans un DataFrame Pandas
```

### 2️⃣ **Prepare Train/Test** (`prepare_train_test_data()`)
```python
- Sépare X (features) et y (target)
- Divise: 80% entraînement, 20% test
- Remplace NaN/Inf par 0
- Affiche distribution des classes
```

### 3️⃣ **Train Models** (`train_models()`)
```python
Pour chaque modèle:
  ├─ GridSearchCV pour trouver les meilleurs paramètres
  ├─ Essaie 50+ combinaisons de paramètres
  ├─ Utilise 5-fold cross-validation
  └─ Garde le meilleur modèle
```

### 4️⃣ **Evaluate** (`evaluate_models()`)
```python
Pour chaque modèle entraîné:
  ├─ Fait des prédictions sur le test set
  ├─ Calcule: Accuracy, Precision, Recall, AUC, F1
  ├─ Crée matrice de confusion
  └─ Affiche un rapport détaillé
```

### 5️⃣ **Save** (`save_models()`)
```python
- Sauvegarde modèles dans .pkl
- Sauvegarde l'importance des features
- Crée un fichier résumé
```

---

## ⚙️ PARAMÈTRES À MODIFIER

Dans `run_training.py`, tu peux changer:

```python
# 1. Quel fichier de données utiliser?
DATA_PATH = "..."  # Chemin à ta donnée nettoyée

# 2. Train/Test split
TEST_SIZE = 0.2  # De 0 à 1 (plus grand = plus de données pour test)

# 3. Colonne cible
TARGET_COLUMN = "failure_within_24h"  # Doit être 0 ou 1

# 4. Modèles à entraîner (None = tous)
MODELS_TO_TRAIN = ["random_forest", "xgboost"]  # ou None

# 5. Validation croisée
CV = 5  # Nombre de folds (5 = classique)

# 6. Graine aléatoire
RANDOM_STATE = 42  # Pour reproductibilité
```

---

## 🚨 Erreurs Courantes et Solutions

### ❌ "FileNotFoundError: No such file"
**Solution:** Vérifiez que DATA_PATH pointe vers un fichier qui existe.
```python
# Lisez d'abord votre dossier
import os
print(os.listdir("data/processed/cleaned_data/"))
```

### ❌ "KeyError: 'failure_within_24h'"
**Solution:** La colonne cible n'existe pas. Changez TARGET_COLUMN:
```python
# Vérifiez les colonnes disponibles
df = pd.read_csv(DATA_PATH)
print(df.columns.tolist())
```

### ❌ "ValueError: Invalid parameter..."
**Solution:** Un paramètre n'est pas valide. Vérifiez MODELS_TO_TRAIN:
```python
# Modèles valides:
["random_forest", "gradient_boosting", "logistic_regression", "xgboost", "lightgbm"]
```

### ❌ "MemoryError" - pas assez de mémoire
**Solution:** Réduisez TEST_SIZE ou utilisez une partie des données.

---

## 📈 Interpréter les Résultats

Après l'entraînement, tu verras:

```
RÉSUMÉ DES PERFORMANCES:
--------------------------------------------------

RANDOM_FOREST
  Accuracy: 0.8932  ← Combien de prédictions justes (0-1)
  AUC:      0.9234  ← Meilleur = 1.0

XGBOOST
  Accuracy: 0.9045
  AUC:      0.9456

...
```

**Guide d'interprétation:**
- **Accuracy > 0.85** = Bon ✅
- **AUC > 0.85** = Bon ✅
- **AUC > 0.95** = Excellent 🌟

---

## 🎯 Prochaines Étapes

1. **Entrainez les modèles** avec `run_training.py`
2. **Utilisez `predict_model.py`** pour faire des prédictions sur nouvelles données
3. **Trackez les performances** avec `monitoring/performance_tracking.py`
4. **Détectez la dérive de données** avec `monitoring/data_drift.py`

---

## 💡 Conseils Pro

✅ **Commencez simple:** Entraînez d'abord Random Forest (rapide, bon)
✅ **Testez ensemble:** Puis XGBoost et LightGBM (plus lents, meilleures perf)
✅ **Sauvegardez tout:** Les modèles sont dans `src/models/models/`
✅ **Comparez:** L'importance des features aide à comprendre les prédictions

---

**C'est prêt! Bon entraînement! 🚀**
