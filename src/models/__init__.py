"""
Package de Prédiction de Risque de Défaillance Industrielle

Ce package contient les modules nécessaires pour entraîner, évaluer et 
utiliser un modèle de prédiction des risques de défaillance industrielle.
"""

from .train_model import train_and_evaluate

__all__ = [
    'train_and_evaluate',
]

# Version du package
__version__ = '0.1.0'

# Informations sur le projet
__project_name__ = 'Prédiction de Risque de Défaillance Industrielle'
__author__ = 'Classe de Machine Learning'
