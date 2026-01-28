#!/usr/bin/env python
"""
ENTRAÎNEMENT INTERACTIF - Choisir votre vitesse d'entraînement
Lance le training avec les paramètres optimaux pour vous
"""

import os
import sys
import subprocess
import time


def show_menu():
    """Afficher le menu de sélection"""
    print("\n" + "="*70)
    print("⚡ ENTRAÎNEMENT INTERACTIF")
    print("="*70 + "\n")
    
    print("Choisissez votre profil d'entraînement:\n")
    
    profiles = {
        "1": {
            "name": "⚡⚡⚡ ULTRA-RAPIDE (1-2 min)",
            "models": ["logistic_regression"],
            "cv": 2,
            "description": "Test rapide avec Logistic Regression"
        },
        "2": {
            "name": "⚡⚡ RAPIDE (3-5 min) ✅ RECOMMANDÉ",
            "models": ["xgboost", "lightgbm", "logistic_regression"],
            "cv": 3,
            "description": "3 modèles rapides avec optimisations"
        },
        "3": {
            "name": "⚡ NORMAL (8-15 min)",
            "models": ["xgboost", "lightgbm", "gradient_boosting", "logistic_regression"],
            "cv": 3,
            "description": "4 modèles pour production"
        },
        "4": {
            "name": "🔬 COMPLET (20-40 min)",
            "models": None,
            "cv": 5,
            "description": "Tous les modèles avec 5-fold CV"
        },
        "5": {
            "name": "🎯 CUSTOM",
            "models": None,
            "cv": None,
            "description": "Configurer manuellement"
        }
    }
    
    for key, profile in profiles.items():
        print(f"{key}. {profile['name']}")
        print(f"   └─ {profile['description']}\n")
    
    return profiles


def get_choice():
    """Obtenir le choix de l'utilisateur"""
    while True:
        choice = input("Sélectionnez (1-5): ").strip()
        if choice in ["1", "2", "3", "4", "5"]:
            return choice
        else:
            print("❌ Choix invalide. Essayez à nouveau.")


def generate_config(profile):
    """Générer la configuration Python"""
    models = profile["models"]
    cv = profile["cv"]
    
    if models:
        models_str = f"[{', '.join([f\"'{m}\" for m in models])}]"
    else:
        models_str = "None"
    
    return models_str, cv


def update_run_training(models_str, cv):
    """Mettre à jour run_training.py"""
    filepath = "run_training.py"
    
    # Lire le fichier
    with open(filepath, "r") as f:
        content = f.read()
    
    # Remplacer les paramètres
    # Chercher la ligne avec MODELS_TO_TRAIN
    import re
    
    content = re.sub(
        r'MODELS_TO_TRAIN = .*',
        f'MODELS_TO_TRAIN = {models_str}',
        content
    )
    
    content = re.sub(
        r'CV = \d+',
        f'CV = {cv}',
        content
    )
    
    # Écrire le fichier
    with open(filepath, "w") as f:
        f.write(content)
    
    return True


def run_training():
    """Lancer le training"""
    print("\n" + "="*70)
    print("🚀 LANCEMENT DE L'ENTRAÎNEMENT...")
    print("="*70 + "\n")
    
    try:
        # Lancer run_training.py
        start_time = time.time()
        
        result = subprocess.run(
            [sys.executable, "run_training.py"],
            cwd=os.getcwd()
        )
        
        elapsed = time.time() - start_time
        
        if result.returncode == 0:
            print("\n" + "="*70)
            print(f"✅ ENTRAÎNEMENT RÉUSSI EN {elapsed/60:.2f} min!")
            print("="*70 + "\n")
            return True
        else:
            print("\n❌ Erreur lors de l'entraînement")
            return False
    
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        return False


def main():
    """Menu principal"""
    try:
        profiles = show_menu()
        choice = get_choice()
        profile = profiles[choice]
        
        if choice == "5":
            # Mode custom
            print("\n🎯 Mode CUSTOM - Configuration manuelle\n")
            
            models_input = input("Modèles (séparés par virgule ou None pour tous): ").strip()
            if models_input.lower() == "none":
                models_str = "None"
            else:
                models = [m.strip() for m in models_input.split(",")]
                models_str = f"[{', '.join([f\"'{m}\" for m in models])}]"
            
            cv = int(input("CV folds (ex: 3 ou 5): ").strip())
            
        else:
            # Profil prédéfini
            models_str, cv = generate_config(profile)
            print(f"\n✅ Profil sélectionné: {profile['name']}")
            print(f"   Modèles: {models_str}")
            print(f"   CV: {cv}\n")
        
        # Mettre à jour run_training.py
        print("📝 Mise à jour de run_training.py...")
        update_run_training(models_str, cv)
        print("✅ Configuration mise à jour\n")
        
        # Lancer le training
        confirm = input("Lancer l'entraînement maintenant? (y/n): ").strip().lower()
        if confirm == "y":
            run_training()
        else:
            print("\n💡 Pour lancer ultérieurement: python run_training.py")
    
    except KeyboardInterrupt:
        print("\n\n❌ Entraînement annulé")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
