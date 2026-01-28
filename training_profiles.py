"""
PROFILS D'ENTRAÎNEMENT - Choisissez votre vitesse
Éditez run_training.py pour utiliser ces profils
"""

# ============================================================
# PROFIL 1: ⚡⚡⚡ ULTRA-RAPIDE (1-2 min)
# ============================================================
ULTRA_FAST = {
    "MODELS_TO_TRAIN": ["logistic_regression"],  # Un seul modèle ultra-rapide
    "CV": 2,  # 2-fold au lieu de 5
    "TEST_SIZE": 0.3,  # Plus de données de test = moins de train
    "DESCRIPTION": "Ultra-rapide: Logistic Regression seul, 2-fold CV"
}


# ============================================================
# PROFIL 2: ⚡⚡ RAPIDE (3-5 min) - RECOMMANDÉ
# ============================================================
FAST = {
    "MODELS_TO_TRAIN": ["xgboost", "lightgbm", "logistic_regression"],
    "CV": 3,  # 3-fold au lieu de 5
    "TEST_SIZE": 0.2,
    "DESCRIPTION": "Rapide: 3 modèles rapides, 3-fold CV"
}


# ============================================================
# PROFIL 3: ⚡ NORMAL (8-15 min)
# ============================================================
NORMAL = {
    "MODELS_TO_TRAIN": ["xgboost", "lightgbm", "gradient_boosting", "logistic_regression"],
    "CV": 3,
    "TEST_SIZE": 0.2,
    "DESCRIPTION": "Normal: 4 modèles, 3-fold CV"
}


# ============================================================
# PROFIL 4: 🔬 COMPLET (20-40 min) - MEILLEURE PERFORMANCE
# ============================================================
COMPLETE = {
    "MODELS_TO_TRAIN": None,  # Tous les modèles
    "CV": 5,  # 5-fold cross-validation
    "TEST_SIZE": 0.2,
    "DESCRIPTION": "Complet: Tous les modèles, 5-fold CV, meilleure performance"
}


# ============================================================
# PROFIL 5: 🎯 CUSTOM - À PERSONNALISER
# ============================================================
CUSTOM = {
    "MODELS_TO_TRAIN": ["xgboost"],  # À modifier
    "CV": 3,  # À modifier
    "TEST_SIZE": 0.2,  # À modifier
    "DESCRIPTION": "Custom: À personnaliser selon vos besoins"
}


# ============================================================
# COMMENT UTILISER
# ============================================================

"""
1. Ouvrir run_training.py
2. Rechercher "CV = " (ligne ~47)
3. Remplacer par votre profil:

   # OPTION 1: Ultra-rapide
   MODELS_TO_TRAIN = ["logistic_regression"]
   CV = 2
   
   # OPTION 2: Rapide (recommandé)
   MODELS_TO_TRAIN = ["xgboost", "lightgbm", "logistic_regression"]
   CV = 3
   
   # OPTION 3: Normal
   MODELS_TO_TRAIN = ["xgboost", "lightgbm", "gradient_boosting", "logistic_regression"]
   CV = 3
   
   # OPTION 4: Complet
   MODELS_TO_TRAIN = None
   CV = 5

4. Sauvegarder et lancer: python run_training.py
"""


# ============================================================
# COMPARAISON
# ============================================================

COMPARISON = """
╔══════════════════════════════════════════════════════════════════╗
║                PROFILS D'ENTRAÎNEMENT - COMPARAISON              ║
╠══════════════════════════════════════════════════════════════════╣
║                                                                  ║
║ ⚡⚡⚡ ULTRA-RAPIDE (1-2 min)                                    ║
║   └─ Logistic Regression seul                                   ║
║   └─ 2-fold CV                                                  ║
║   └─ Idéal pour: tester rapidement                              ║
║   └─ Performance: Moyenne ★★☆☆☆                               ║
║                                                                  ║
║ ⚡⚡ RAPIDE (3-5 min) ✅ RECOMMANDÉ                              ║
║   └─ XGBoost + LightGBM + Logistic Regression                   ║
║   └─ 3-fold CV                                                  ║
║   └─ Idéal pour: développement, itération rapide                ║
║   └─ Performance: Bonne ★★★★☆                                  ║
║                                                                  ║
║ ⚡ NORMAL (8-15 min)                                             ║
║   └─ 4 modèles sans Random Forest                               ║
║   └─ 3-fold CV                                                  ║
║   └─ Idéal pour: production                                      ║
║   └─ Performance: Très bonne ★★★★☆                             ║
║                                                                  ║
║ 🔬 COMPLET (20-40 min)                                           ║
║   └─ Tous les modèles                                           ║
║   └─ 5-fold CV                                                  ║
║   └─ Idéal pour: études complètes                               ║
║   └─ Performance: Excellente ★★★★★                             ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝
"""


# ============================================================
# AFFICHAGE DES PROFILS
# ============================================================

def show_profiles():
    """Afficher les profils disponibles"""
    print(COMPARISON)
    print("\nProfils disponibles:")
    for name, config in [
        ("ULTRA_FAST", ULTRA_FAST),
        ("FAST", FAST),
        ("NORMAL", NORMAL),
        ("COMPLETE", COMPLETE),
    ]:
        print(f"\n{name}:")
        print(f"  Description: {config['DESCRIPTION']}")
        print(f"  Modèles: {config['MODELS_TO_TRAIN']}")
        print(f"  CV: {config['CV']}")


def apply_profile(profile_name):
    """Appliquer un profil"""
    profiles = {
        "ultra_fast": ULTRA_FAST,
        "fast": FAST,
        "normal": NORMAL,
        "complete": COMPLETE,
    }
    
    if profile_name in profiles:
        config = profiles[profile_name]
        print(f"\n✅ Profil '{profile_name}' sélectionné")
        print(f"   {config['DESCRIPTION']}")
        return config
    else:
        print(f"❌ Profil '{profile_name}' non trouvé")
        return None


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        profile = apply_profile(sys.argv[1])
    else:
        show_profiles()
