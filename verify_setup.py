#!/usr/bin/env python
"""
Script de test pour vérifier que tout est prêt pour l'entraînement.
Exécutez: python verify_setup.py
"""

import os
import sys
from pathlib import Path

def check_file_exists(path, description):
    """Vérifier qu'un fichier existe"""
    if os.path.exists(path):
        print(f"✅ {description}: {path}")
        return True
    else:
        print(f"❌ {description}: NOT FOUND - {path}")
        return False

def check_module_importable(module_name, description):
    """Vérifier qu'un module Python peut être importé"""
    try:
        __import__(module_name)
        print(f"✅ {description}: {module_name}")
        return True
    except ImportError:
        print(f"❌ {description}: NOT INSTALLED - pip install {module_name}")
        return False

def check_csv_has_column(csv_path, column_name):
    """Vérifier qu'un CSV a une colonne"""
    try:
        import pandas as pd
        df = pd.read_csv(csv_path)
        if column_name in df.columns:
            print(f"✅ CSV a la colonne '{column_name}'")
            return True
        else:
            print(f"❌ CSV n'a PAS la colonne '{column_name}'")
            print(f"   Colonnes disponibles: {df.columns.tolist()}")
            return False
    except Exception as e:
        print(f"❌ Erreur lors de la lecture du CSV: {e}")
        return False

def main():
    print("\n" + "="*60)
    print("VÉRIFICATION DE LA CONFIGURATION")
    print("="*60 + "\n")
    
    all_ok = True
    
    # 1. Vérifier les modules Python
    print("1️⃣  DÉPENDANCES PYTHON:")
    print("-" * 60)
    deps = [
        ("pandas", "Pandas"),
        ("numpy", "NumPy"),
        ("sklearn", "Scikit-learn"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
        ("joblib", "Joblib"),
    ]
    
    for module, name in deps:
        if not check_module_importable(module, name):
            all_ok = False
    
    # 2. Vérifier les fichiers de code
    print("\n2️⃣  FICHIERS DE CODE:")
    print("-" * 60)
    
    files_to_check = [
        ("src/data/extract.py", "Extract module"),
        ("src/data/clean.py", "Clean module"),
        ("src/features/build_features.py", "Features module"),
        ("src/models/train_model.py", "Train module (⭐ PRINCIPAL)"),
        ("src/models/predict_model.py", "Predict module"),
        ("run_training.py", "Run training script"),
    ]
    
    for filepath, description in files_to_check:
        if not check_file_exists(filepath, description):
            all_ok = False
    
    # 3. Vérifier les répertoires
    print("\n3️⃣  RÉPERTOIRES:")
    print("-" * 60)
    
    dirs_to_check = [
        ("data/raw", "Dossier données brutes"),
        ("data/processed/cleaned_data", "Dossier données nettoyées"),
        ("src/models/models", "Dossier modèles"),
    ]
    
    for dirpath, description in dirs_to_check:
        if os.path.isdir(dirpath):
            print(f"✅ {description}: {dirpath}")
        else:
            print(f"⚠️  {description}: NOT FOUND - {dirpath} (sera créé)")
    
    # 4. Vérifier les données
    print("\n4️⃣  DONNÉES NETTOYÉES:")
    print("-" * 60)
    
    cleaned_dir = "data/processed/cleaned_data"
    if os.path.isdir(cleaned_dir):
        csv_files = [f for f in os.listdir(cleaned_dir) if f.endswith('.csv')]
        if csv_files:
            print(f"✅ CSV trouvés dans {cleaned_dir}:")
            for csv_file in csv_files:
                print(f"   - {csv_file}")
                
                # Vérifier la colonne cible
                csv_path = os.path.join(cleaned_dir, csv_file)
                if not check_csv_has_column(csv_path, "failure_within_24h"):
                    all_ok = False
        else:
            print(f"❌ Aucun fichier CSV dans {cleaned_dir}")
            print("   Exécutez d'abord: python src/data/extract.py")
            print("   Puis:             python src/data/clean.py")
            all_ok = False
    else:
        print(f"❌ Dossier {cleaned_dir} n'existe pas")
        all_ok = False
    
    # 5. Vérifier les corrections apportées
    print("\n5️⃣  VÉRIFICATION DES CORRECTIONS:")
    print("-" * 60)
    
    try:
        with open("src/models/train_model.py", 'r') as f:
            content = f.read()
            
        # Vérifier que les corrections ont été appliquées
        checks = [
            ("'random_forest': {", "✅ Random Forest modèle ajouté"),
            ("'xgboost': {", "✅ XGBoost modèle ajouté"),
            ("accuracy = (y_pred == y_test)", "✅ Accuracy calculée correctement"),
            ("auc_score = roc_auc_score", "✅ AUC calculée correctement"),
            ("def train_and_evaluate(", "✅ Fonction train_and_evaluate() existe"),
            ("def find_best_model(", "✅ Fonction find_best_model() existe"),
        ]
        
        for check_string, message in checks:
            if check_string in content:
                print(message)
            else:
                print(f"❌ Correction manquante: {message}")
                all_ok = False
                
    except Exception as e:
        print(f"❌ Erreur lors de la vérification du code: {e}")
        all_ok = False
    
    # Résumé final
    print("\n" + "="*60)
    if all_ok:
        print("✅ TOUT EST OK! Vous pouvez lancer:")
        print("   python run_training.py")
    else:
        print("❌ ERREURS DÉTECTÉES")
        print("\n   Corrigez les erreurs ci-dessus, puis réessayez.")
        print("\n   Étapes recommandées:")
        print("   1. pip install pandas numpy scikit-learn xgboost lightgbm joblib")
        print("   2. python src/data/extract.py")
        print("   3. python src/data/clean.py")
        print("   4. Modifier DATA_PATH dans run_training.py")
        print("   5. python run_training.py")
    print("="*60 + "\n")
    
    return 0 if all_ok else 1

if __name__ == "__main__":
    sys.exit(main())
