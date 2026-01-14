# Configuration Conda pour le Projet ML

## 📋 Prérequis

- **Conda** (Miniconda ou Anaconda) - [Télécharger ici](https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html)
- **Windows 10/11** ou **Linux/Mac**

## 🚀 Installation Rapide

### Option 1: Installation Automatique (Recommandée)

#### Sur Windows:
```powershell
# Ouvrir PowerShell dans le dossier du projet, puis:
.\setup_conda.bat
```

#### Sur Linux/Mac:
```bash
chmod +x setup_conda.sh
./setup_conda.sh
```

### Option 2: Installation Manuelle

#### Étape 1: Créer l'environnement
```bash
conda env create -f environment.yml
```

#### Étape 2: Activer l'environnement
```bash
conda activate ml-predictive-maintenance
```

#### Étape 3: Vérifier l'installation
```bash
python verify_setup.py
```

## 📦 Environnement Conda Créé

L'environnement `ml-predictive-maintenance` contient:

| Package | Version | Description |
|---------|---------|-------------|
| Python | 3.10 | Langage de programmation |
| NumPy | Latest | Calcul numérique |
| Pandas | Latest | Manipulation de données |
| Scikit-learn | Latest | Machine Learning |
| Matplotlib | Latest | Visualisation |
| Seaborn | Latest | Graphiques statistiques |
| SciPy | Latest | Calcul scientifique |
| XGBoost | 2.0.3 | Gradient Boosting |
| LightGBM | 4.1.1 | Gradient Boosting léger |
| Joblib | 1.3.2 | Sérialisation |
| Weights & Biases | 0.16.1 | Suivi des expériences |

## 🔧 Commandes Utiles

### Activer l'environnement
```bash
conda activate ml-predictive-maintenance
```

### Désactiver l'environnement
```bash
conda deactivate
```

### Lister tous les environnements
```bash
conda env list
```

### Supprimer l'environnement (si nécessaire)
```bash
conda env remove -n ml-predictive-maintenance
```

### Mettre à jour les packages
```bash
conda activate ml-predictive-maintenance
conda update --all
```

## 📂 Structure des Répertoires

Assurez-vous que cette structure existe avant de lancer l'entraînement:

```
.
├── data/
│   ├── raw/
│   │   ├── predictive_maintenance_sensor_data.csv
│   │   └── predictive_maintenance_failure_logs.csv
│   ├── processed/
│   │   ├── cleaned_data/
│   │   ├── augmented_data/
│   │   └── extracted_data/
│   └── validation/
├── src/
│   ├── data/
│   ├── features/
│   ├── models/
│   └── monitoring/
├── tests/
├── run_training.py
└── verify_setup.py
```

## 🎯 Lancer l'Entraînement

Une fois l'environnement activé:

```bash
conda activate ml-predictive-maintenance
python run_training.py
```

## ✅ Vérification

Pour vérifier que tout est bien configuré:

```bash
conda activate ml-predictive-maintenance
python verify_setup.py
```

Vous devriez voir:
- ✅ Toutes les dépendances Python installées
- ✅ Tous les fichiers de code présents
- ✅ Tous les répertoires créés

## 🐛 Dépannage

### Erreur: "conda: command not found"
- **Solution**: Réinstallez Conda ou ajoutez-le au PATH de votre système

### Erreur: "Failed to install packages"
- **Solution**: Mettez à jour conda: `conda update conda`

### Erreur: "Module not found"
- **Solution**: Vérifiez que l'environnement est activé: `conda activate ml-predictive-maintenance`

### Les packages GPU ne se chargent pas
- **Solution**: Les versions CPU/GPU sont configurées automatiquement. Si vous avez une GPU NVIDIA, assurez-vous d'avoir CUDA installé.

## 📞 Support

Pour plus d'informations sur conda, consultez:
- [Documentation Conda](https://docs.conda.io/)
- [Conda Cheat Sheet](https://docs.conda.io/projects/conda/en/latest/user-guide/cheatsheet.html)
