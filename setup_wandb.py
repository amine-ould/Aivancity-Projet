#!/usr/bin/env python
"""
Script pour initialiser et configurer WandB.
Usage: python setup_wandb.py
"""

import os
import sys
import subprocess
import getpass


def check_wandb_installed():
    """Vérifier si wandb est installé"""
    try:
        import wandb
        print(f"✅ WandB {wandb.__version__} est installé")
        return True
    except ImportError:
        print("❌ WandB n'est pas installé")
        return False


def install_wandb():
    """Installer wandb"""
    print("\n📦 Installation de WandB...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "wandb"])
        print("✅ WandB installé avec succès")
        return True
    except subprocess.CalledProcessError:
        print("❌ Erreur lors de l'installation de WandB")
        return False


def check_wandb_login():
    """Vérifier si l'utilisateur est authentifié"""
    import wandb
    
    try:
        # Essayer d'accéder à la clé API
        api = wandb.Api()
        user = api.default_entity
        print(f"✅ Connecté en tant que: {user}")
        return True
    except:
        print("❌ Non authentifié à WandB")
        return False


def login_wandb():
    """Authentifier l'utilisateur avec WandB"""
    print("\n🔐 Authentification WandB")
    print("Allez sur: https://wandb.ai/authorize")
    print("\n1. Connectez-vous avec votre compte WandB (ou créez-en un gratuit)")
    print("2. Copiez votre API Key")
    print("3. Collez-la ci-dessous (elle sera masquée)\n")
    
    api_key = getpass.getpass("Entrez votre API Key: ").strip()
    
    if api_key:
        os.environ["WANDB_API_KEY"] = api_key
        print("✅ API Key définie")
        return True
    else:
        print("❌ Pas d'API Key fournie")
        return False


def test_wandb():
    """Tester la connexion WandB"""
    print("\n🧪 Test de connexion...\n")
    
    try:
        import wandb
        
        # Initialiser un run de test
        run = wandb.init(
            project="test-integration",
            name="test-run",
            reinit=True
        )
        
        # Enregistrer une métrique
        wandb.log({"test_metric": 42})
        
        # Terminer
        wandb.finish()
        
        print(f"✅ Connexion réussie!")
        return True
        
    except Exception as e:
        print(f"❌ Erreur lors du test: {e}")
        return False


def show_setup_info():
    """Afficher les informations de configuration"""
    print("\n" + "="*60)
    print("📊 CONFIGURATION WANDB")
    print("="*60)
    print("\n✅ Maintenant vous pouvez:")
    print("  1. Éditer WANDB_CONFIG dans run_training.py")
    print("  2. Lancer: python run_training.py")
    print("  3. Accéder aux résultats sur: https://wandb.ai/\n")
    print("💡 Conseils:")
    print("  - Changez 'project' pour organiser vos expériences")
    print("  - Utilisez 'tags' pour filtrer les runs")
    print("  - Ajoutez des 'notes' pour documenter")
    print("="*60 + "\n")


def main():
    """Flux principal"""
    print("\n" + "="*60)
    print("🎯 SETUP WANDB")
    print("="*60 + "\n")
    
    # 1. Vérifier l'installation
    print("1️⃣  Vérification de l'installation...")
    if not check_wandb_installed():
        print("   Installation en cours...")
        if not install_wandb():
            sys.exit(1)
    
    # 2. Vérifier l'authentification
    print("\n2️⃣  Vérification de l'authentification...")
    if not check_wandb_login():
        print("   Authentification requise...")
        if not login_wandb():
            print("   Vous pouvez vous authentifier plus tard avec: wandb login")
            sys.exit(1)
    
    # 3. Tester la connexion
    print("\n3️⃣  Test de la connexion...")
    if not test_wandb():
        print("⚠️  Le test a échoué, mais vous pouvez quand même continuer")
    
    # 4. Afficher les infos
    show_setup_info()


if __name__ == "__main__":
    try:
        main()
        print("✅ Setup terminé avec succès!")
    except KeyboardInterrupt:
        print("\n❌ Setup annulé par l'utilisateur")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Erreur: {e}")
        sys.exit(1)
