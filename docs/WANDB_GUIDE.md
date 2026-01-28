# Guide W&B (suivi des expérimentations) — Détails complets

## 1) Pourquoi W&B dans ce projet
W&B sert à :
- **Tracer les métriques** (accuracy, AUC, F1, etc.)
- **Comparer les runs** (modèles, paramètres, datasets)
- **Stocker les artefacts** (modèles .pkl, datasets, graphiques)
- **Collaborer** (partage de dashboards)

Dans ce projet, l’intégration est **déjà active** dans `run_training.py`.

---

## 2) Fichiers W&B utilisés
- `run_training.py` : initialise W&B et loggue les métriques
- `wandb/wandb_config.py` : charge la clé depuis `.env.wandb`
- `wandb/wandb_metrics_logger.py` : logs avancés (figures, tables, artefacts)
- `wandb/wandb_tools.py` : utilitaires (cleanup, sync offline)
- `wandb/wandb_sweeps.py` : hyperparameter tuning
- `wandb/wandb_templates.py` : templates d’intégration

---

## 3) Installation et authentification

### 3.1 Installer W&B
```bash
pip install wandb
```

### 3.2 Authentifier
```bash
python setup_wandb.py
```
ou
```bash
wandb login
```

---

## 4) Configuration de W&B

### 4.1 Config principale dans `run_training.py`
```python
WANDB_CONFIG = {
  "project": "industrial-failure-prediction",
  "entity": "<ton_workspace>",
  "enable_wandb": True,
  "tags": ["training"],
  "notes": "Run d’entrainement"
}
```

### 4.2 Variables d’environnement
Créer un fichier `.env.wandb` à la racine :
```
WANDB_API_KEY=ta_cle_api
```

Ou exporter la variable système :
```
WANDB_API_KEY=ta_cle_api
```

---

## 5) Ce qui est loggué dans ce projet
Quand `enable_wandb = True`, le script :
- initialise un **run**
- loggue des **métriques** (accuracy, auc, etc.)
- loggue des **tables** (samples dataset)
- loggue des **figures** (heatmap, distribution)
- crée des **artefacts** (dataset + modèles)

---

## 6) Mode offline / online

### 6.1 Offline
Pour travailler sans internet :
```
WANDB_MODE=offline
```

### 6.2 Synchroniser plus tard
```
wandb sync wandb/
```

---

## 7) Sweeps (tuning automatique)
Les sweeps permettent d’explorer automatiquement des hyperparamètres.

### Lancer un sweep
```bash
python wandb/wandb_sweeps.py
```

Le script :
- crée un sweep (grid / random / bayesian)
- lance des agents
- compare les runs

---

## 8) Outils avancés

### 8.1 Templates d’intégration
```bash
python wandb/wandb_templates.py
```
Affiche des exemples prêts à l’emploi pour :
- classification
- régression
- tracking avancé

### 8.2 Tools (maintenance)
```bash
python wandb/wandb_tools.py
```
Permet :
- de nettoyer le cache
- de synchroniser les runs offline
- de diagnostiquer l’environnement

---

## 9) Dépannage complet

### ImportError: wandb
```bash
pip install wandb
```

### Not authenticated
```bash
python setup_wandb.py
```

### Runs non visibles
- vérifier `WANDB_MODE`
- vérifier `WANDB_API_KEY`

### Problèmes de permissions
- vérifier `entity` dans `WANDB_CONFIG`
- s’assurer que le projet existe sur WandB

---

## 10) Bonnes pratiques
- Nommer clairement les runs
- Logguer les hyperparamètres en début de run
- Sauvegarder les meilleurs modèles en artefacts
- Utiliser des tags pour filtrer rapidement

---

Si tu veux une configuration avancée (multi‑projets, teams, artefacts versionnés), dis‑moi et j’ajoute une section dédiée.
