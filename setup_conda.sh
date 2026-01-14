#!/bin/bash
# Script de configuration de l'environnement Conda pour le projet ML

echo "========================================================"
echo "Configuration de l'environnement Conda"
echo "========================================================"

# 1. Créer l'environnement conda
echo ""
echo "[1/3] Création de l'environnement conda..."
echo ""
conda env create -f environment.yml

# 2. Activer l'environnement
echo ""
echo "[2/3] Activation de l'environnement..."
echo ""
source activate ml-predictive-maintenance

# 3. Vérifier l'installation
echo ""
echo "[3/3] Vérification de l'installation..."
echo ""
python verify_setup.py

echo ""
echo "========================================================"
echo "Configuration terminée !"
echo "========================================================"
echo ""
echo "Pour activer l'environnement à l'avenir, utilisez:"
echo "  conda activate ml-predictive-maintenance"
echo ""
echo "Pour lancer l'entraînement:"
echo "  python run_training.py"
echo ""
