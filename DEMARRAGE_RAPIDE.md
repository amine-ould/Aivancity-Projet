## 🚀 DÉMARRAGE RAPIDE EN 3 ÉTAPES

### ✅ ÉTAPE 1: Préparez vos données

Avant d'entraîner, assurez-vous d'avoir un fichier CSV nettoyé:

```bash
# Exécutez dans cet ordre:
python src/data/extract.py    # Extraire CSV bruts
python src/data/clean.py      # Nettoyer les données
```

Le fichier doit avoir:
- Une colonne `failure_within_24h` (0 ou 1) = ce que vous prédisez
- Colonnes de features: temperature, vibration, pressure, current, etc.

### ✅ ÉTAPE 2: Modifiez run_training.py

Ouvrez `run_training.py` et changez 1 ligne:

```python
# AVANT:
DATA_PATH = r"data\processed\cleaned_data\sensor_data_cleaned.csv"

# APRÈS - remplacer par VOTRE fichier:
DATA_PATH = r"chemin/vers/votre/cleaned_data.csv"
```

Trouvez votre fichier:
```python
import os
files = os.listdir("data/processed/cleaned_data/")
print(files)  # Affiche les fichiers disponibles
```

### ✅ ÉTAPE 3: Lancez l'entraînement

```bash
python run_training.py
```

C'est tout! ⏱️ Ça prendra 10-30 minutes selon votre machine.

---

## 📊 Résultats

Après l'entraînement:

```
✅ Modèles sauvegardés dans: src/models/models/
   ├─ random_forest_20250114_143022.pkl
   ├─ xgboost_20250114_143022.pkl
   ├─ random_forest_feature_importance_20250114_143022.csv
   └─ training_summary_20250114_143022.pkl

📈 Vous verrez aussi une table de performances:
   RANDOM FOREST: Accuracy=0.8932, AUC=0.9234
   XGBOOST:       Accuracy=0.9045, AUC=0.9456
```

---

## ❓ Questions Courantes

**Q: "FileNotFoundError: No such file"**
A: Vérifiez que DATA_PATH est correct. Lisez le dossier:
```python
print(os.listdir("data/processed/cleaned_data/"))
```

**Q: "KeyError: 'failure_within_24h'"**
A: La colonne cible n'existe pas. Vérifiez vos données:
```python
import pandas as pd
df = pd.read_csv(DATA_PATH)
print(df.columns.tolist())
print(df['failure_within_24h'].unique())  # Doit être [0, 1]
```

**Q: Combien de temps ça prend?**
A: 10-30 minutes (5 modèles × GridSearch):
- Random Forest: 2-5 min
- Gradient Boosting: 3-8 min
- Logistic Regression: < 1 min
- XGBoost: 2-4 min
- LightGBM: 1-3 min

**Q: Comment utiliser le modèle après?**
A: Utilisez `predict_model.py`:
```python
from src.models.predict_model import PredictionEngine

engine = PredictionEngine(model_path="src/models/models/xgboost_*.pkl")
predictions = engine.predict(new_data)
```

**Q: Puis-je ne former que certains modèles?**
A: Oui, modifiez `run_training.py`:
```python
MODELS_TO_TRAIN = ["random_forest", "xgboost"]  # Au lieu de None
```

---

## 📖 Pour Plus de Détails

Lisez ces fichiers:
- `EXPLICATION_DETAILLEE.txt` - Explication complète
- `GUIDE_ENTRAÎNEMENT.md` - Guide détaillé avec tous les paramètres
- `CHECKLIST.md` - Liste de vérification avant lancement

---

**Besoin d'aide?** 📧 Consultez la documentation!
