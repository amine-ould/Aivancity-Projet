"""
WANDB QUICK REFERENCE CARD
Copier-coller des commandes et code snippets
"""

# ============================================================
# COMMANDES TERMINAL
# ============================================================

COMMANDS = """
# INSTALLATION & AUTHENTIFICATION
pip install wandb
python setup_wandb.py              # Automatique
wandb login                        # Manuel

# LANCER L'ENTRAÎNEMENT
python run_training.py             # Avec WandB intégré
python run_training_fast.py        # Version rapide

# OUTILS
python wandb/wandb_tools.py         # Menu d'outils
python wandb/wandb_sweeps.py        # Hyperparameter tuning
python wandb/wandb_templates.py     # Voir les templates

# SYNCHRONISER OFFLINE
wandb sync wandb/                  # Synchroniser les données

# DÉSACTIVER WANDB (si besoin)
export WANDB_DISABLED=true         # Linux/Mac
set WANDB_DISABLED=true            # Windows

# RESET
wandb offline                      # Mode offline
wandb online                       # Mode online
"""

# ============================================================
# CODE SNIPPETS
# ============================================================

SNIPPETS = {
    "init_basic": """
import wandb
wandb.init(project="my-project")
wandb.log({"accuracy": 0.95})
wandb.finish()
""",

    "init_with_config": """
import wandb
wandb.init(
    project="my-project",
    config={"lr": 0.01, "epochs": 100}
)
wandb.log({"loss": 0.05})
wandb.finish()
""",

    "log_metrics": """
wandb.log({
    "accuracy": 0.95,
    "auc": 0.92,
    "f1": 0.88,
    "epoch": 1
})
""",

    "log_model": """
import joblib
joblib.dump(model, "model.pkl")

artifact = wandb.Artifact("my_model", type="model")
artifact.add_file("model.pkl")
wandb.log_artifact(artifact)
""",

    "log_image": """
import matplotlib.pyplot as plt
fig, ax = plt.subplots()
ax.plot([1, 2, 3])
wandb.log({"plot": wandb.Image(fig)})
plt.close()
""",

    "compare_models": """
for model_name in ["rf", "xgb", "lgb"]:
    wandb.init(project="my-project", name=model_name)
    # train...
    wandb.log({"accuracy": score})
    wandb.finish()
""",

    "group_runs": """
wandb.init(
    project="my-project",
    group="hyperparameter-tuning",
    name="exp-001"
)
""",

    "use_helper": """
from wandb.wandb_helper import WandBHelper
helper = WandBHelper()
helper.init(config={"model": "rf"})
helper.log_metrics({"accuracy": 0.95})
helper.log_model("model.pkl", "my_model")
helper.finish()
""",

    "disable_wandb": """
# En développement
wandb.init(mode="disabled")  # Ne rien envoyer

# Ou dans run_training.py
WANDB_CONFIG["enable_wandb"] = False
""",

    "offline_mode": """
wandb.init(mode="offline")
# Les données sont sauvegardées localement
# Synchroniser plus tard avec: wandb sync
""",

    "error_handling": """
try:
    wandb.init(project="my-project")
    # votre code...
    wandb.log({"success": True})
except Exception as e:
    wandb.log({"error": str(e)})
    wandb.finish(exit_code=1)
else:
    wandb.finish(exit_code=0)
""",

    "sweep_config": """
sweep_config = {
    "method": "bayes",
    "metric": {"name": "accuracy", "goal": "maximize"},
    "parameters": {
        "n_estimators": {"values": [100, 200, 300]},
        "max_depth": {"values": [10, 20, 30]}
    }
}
""",
}

# ============================================================
# FICHIERS IMPORTANTES
# ============================================================

FILES = """
🚀 COMMENCER (5 min)
├─ README_WANDB.md
├─ setup_wandb.py
└─ run_training.py

📖 DOCUMENTATION (30-45 min)
├─ GUIDE_WANDB.md
├─ CHECKLIST_WANDB.md
├─ INTEGRATION_WANDB_GUIDE.md
└─ INDEX_WANDB.md

💻 CODE (Prêt à l'emploi)
├─ EXAMPLES_WANDB.py (7 exemples)
├─ wandb/wandb_templates.py (9 templates)
├─ wandb/wandb_helper.py (classe helper)
└─ run_training.py (intégré)

🔧 OUTILS
├─ setup_wandb.py (setup automatique)
├─ wandb/wandb_tools.py (gestion)
├─ wandb/wandb_sweeps.py (hyperparameter tuning)
└─ wandb/wandb_templates.py (voir templates)
"""

# ============================================================
# AFFICHAGE
# ============================================================

def print_quick_reference():
    """Afficher la quick reference"""
    print("\n" + "="*60)
    print("⚡ WANDB QUICK REFERENCE")
    print("="*60 + "\n")
    
    print("COMMANDES ESSENTIELLES:")
    print(COMMANDS)
    
    print("\n" + "="*60)
    print("CODE SNIPPETS:")
    print("="*60 + "\n")
    
    for i, (name, code) in enumerate(SNIPPETS.items(), 1):
        print(f"{i}. {name.upper()}")
        print(code)
        print()
    
    print("="*60)
    print("FICHIERS IMPORTANTS:")
    print("="*60)
    print(FILES)


def print_command_help(cmd):
    """Afficher l'aide pour une commande spécifique"""
    helps = {
        "init": "Initialiser un run WandB",
        "log": "Enregistrer des métriques",
        "finish": "Terminer un run",
        "artifact": "Enregistrer un artefact",
        "sweep": "Créer un hyperparameter sweep",
        "agent": "Lancer un agent sweep",
    }
    
    if cmd in helps:
        print(f"Help: {cmd}")
        print(f"Description: {helps[cmd]}")
    else:
        print(f"Commande '{cmd}' non trouvée")
        print("Commandes disponibles:", list(helps.keys()))


def get_snippet(snippet_name):
    """Retourner un snippet de code"""
    if snippet_name in SNIPPETS:
        print(f"\n{snippet_name}:")
        print(SNIPPETS[snippet_name])
    else:
        print(f"Snippet '{snippet_name}' non trouvé")
        print("Snippets disponibles:")
        for name in SNIPPETS.keys():
            print(f"  - {name}")


# ============================================================
# TABLEAU DE CONFIGURATION
# ============================================================

CONFIG_REFERENCE = """
WANDB_CONFIG = {
    # REQUISES
    "project": "industrial-failure-prediction",  # Nom du projet
    
    # OPTIONNELLES
    "entity": None,               # Username WandB
    "enable_wandb": True,         # Activer/Désactiver
    "tags": [],                   # ["training", "production"]
    "notes": "",                  # Description de l'expérience
    "group": None,                # Pour grouper les runs
    "job_type": None              # "training", "evaluation", etc.
}

# Modifier dans run_training.py (~ligne 30)
"""

# ============================================================
# MÉTRIQUES COURANTES
# ============================================================

COMMON_METRICS = """
CLASSIFICATION
├─ accuracy: Pourcentage de prédictions correctes
├─ precision: TP/(TP+FP)
├─ recall: TP/(TP+FN)
├─ f1: 2 * (precision * recall) / (precision + recall)
├─ auc: Area Under the ROC Curve
└─ confusion_matrix: Vraies/Fausses positifs/négatifs

REGRESSION
├─ mse: Mean Squared Error
├─ mae: Mean Absolute Error
├─ rmse: Root Mean Squared Error
├─ r2: R-squared score
└─ mape: Mean Absolute Percentage Error

TRAINING
├─ loss: Fonction de perte
├─ val_loss: Loss de validation
├─ epoch: Epoch actuelle
└─ learning_rate: Taux d'apprentissage
"""

# ============================================================
# MENU INTERACTIF
# ============================================================

def interactive_menu():
    """Menu interactif"""
    print("\n" + "="*60)
    print("📋 WANDB QUICK REFERENCE - MENU")
    print("="*60 + "\n")
    
    while True:
        print("1. Afficher tous les commands")
        print("2. Voir tous les code snippets")
        print("3. Voir la configuration")
        print("4. Voir les métriques courantes")
        print("5. Afficher les fichiers importants")
        print("6. Quitter")
        
        choice = input("\nChoisissez (1-6): ").strip()
        
        if choice == "1":
            print(COMMANDS)
        elif choice == "2":
            print_quick_reference()
        elif choice == "3":
            print(CONFIG_REFERENCE)
        elif choice == "4":
            print(COMMON_METRICS)
        elif choice == "5":
            print(FILES)
        elif choice == "6":
            break
        else:
            print("Choix invalide")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        if sys.argv[1] == "all":
            print_quick_reference()
        elif sys.argv[1] == "config":
            print(CONFIG_REFERENCE)
        elif sys.argv[1] == "metrics":
            print(COMMON_METRICS)
        elif sys.argv[1] == "files":
            print(FILES)
        else:
            get_snippet(sys.argv[1])
    else:
        interactive_menu()
