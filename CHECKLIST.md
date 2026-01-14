## ✅ CHECKLIST D'ENTRAÎNEMENT

### Avant de lancer l'entraînement:

- [ ] **Vérifier les dépendances:**
  ```bash
  pip install pandas numpy scikit-learn xgboost lightgbm joblib
  ```

- [ ] **Préparer les données:**
  - [ ] Exécuter `python src/data/extract.py` (extraire les CSV bruts)
  - [ ] Exécuter `python src/data/clean.py` (nettoyer les données)
  - [ ] Vérifier qu'un fichier CSV existe dans `data/processed/cleaned_data/`

- [ ] **Configurer `run_training.py`:**
  - [ ] Changer `DATA_PATH` vers le fichier nettoyé
  - [ ] Vérifier que `TARGET_COLUMN = "failure_within_24h"` existe dans vos données
  - [ ] (Optionnel) Sélectionner les modèles avec `MODELS_TO_TRAIN`

- [ ] **Vérifier les permissions:**
  - [ ] Dossier `src/models/models/` accessible en écriture
  - [ ] Dossier `data/` accessible en lecture

### Lancer l'entraînement:

```bash
python run_training.py
```

⏱️ Temps estimé:
- Random Forest: 2-5 min
- Gradient Boosting: 3-8 min
- Logistic Regression: < 1 min
- XGBoost: 2-4 min
- LightGBM: 1-3 min
- **TOTAL: 10-30 minutes** (selon votre machine)

### Après l'entraînement:

- [ ] Vérifier que les fichiers `.pkl` ont été créés dans `src/models/models/`
- [ ] Lire le fichier `GUIDE_ENTRAÎNEMENT.md` pour les prochaines étapes
- [ ] Examiner les `*_feature_importance.csv` pour comprendre les prédictions
- [ ] Utiliser `predict_model.py` pour faire des prédictions

### Si des erreurs:

1. Vérifier que le fichier DATA_PATH existe:
   ```python
   import os
   print(os.path.exists("votre_chemin.csv"))
   ```

2. Vérifier les colonnes:
   ```python
   import pandas as pd
   df = pd.read_csv("votre_chemin.csv")
   print(df.columns.tolist())
   print(df['failure_within_24h'].unique())
   ```

3. Vérifier les importations:
   ```python
   import xgboost
   import lightgbm
   print("✅ Dépendances OK")
   ```

---

**Questions?** Consultez `GUIDE_ENTRAÎNEMENT.md` 📖
