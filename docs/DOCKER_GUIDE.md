docker run --rm -it -v ${PWD}:/app ml-project-sprint
docker run --rm -it -v ${PWD}:/app ml-project-sprint python train_final.py
docker compose -f docker/docker-compose.yml up --build
docker compose -f docker/docker-compose.yml down
# Guide Docker (conteneurisation) — Détails complets

## 1) Pourquoi Docker ici
Docker permet de :
- **reproduire** exactement l’environnement (dépendances + Python)
- **simplifier** l’exécution (une commande unique)
- **déployer** facilement sur une autre machine

---

## 2) Fichiers Docker
- `docker/Dockerfile` : image basée sur micromamba
- `docker/docker-compose.yml` : orchestration simple
- `docker/requirements.txt` : dépendances optionnelles
- `.dockerignore` : exclut les fichiers inutiles

---

## 3) Construire l’image
Depuis la racine :
```bash
docker build -f docker/Dockerfile -t ml-project-sprint .
```

Ce que fait le build :
- copie `environment.yml`
- crée l’environnement conda `ml-predictive-maintenance`
- copie tout le code dans `/app`
- définit la commande par défaut

---

## 4) Lancer un conteneur
```bash
docker run --rm -it -v ${PWD}:/app ml-project-sprint
```

### 4.1 Pourquoi le volume `-v ${PWD}:/app`
- synchronise le code local avec le conteneur
- évite de reconstruire l’image à chaque modification

### 4.2 Lancer un script spécifique
```bash
docker run --rm -it -v ${PWD}:/app ml-project-sprint python train_final.py
```

---

## 5) Docker Compose
```bash
docker compose -f docker/docker-compose.yml up --build
```

Pour arrêter :
```bash
docker compose -f docker/docker-compose.yml down
```

Compose est utile si tu veux :
- standardiser l’exécution
- ajouter d’autres services plus tard

---

## 6) Variables d’environnement W&B
Dans PowerShell :
```
$env:WANDB_API_KEY="votre_cle_api"
$env:WANDB_MODE="online"
```

Puis :
```bash
docker compose -f docker/docker-compose.yml up --build
```

---

## 7) GPU (optionnel)
Si GPU disponible et NVIDIA Toolkit installé :
```bash
docker run --rm -it --gpus all -v ${PWD}:/app ml-project-sprint
```

---

## 8) Modifier la commande par défaut
Dans `docker/Dockerfile` :
```
CMD ["python", "run_training.py"]
```

Tu peux la remplacer par :
- `train_final.py`
- `run_training_fast.py`
- `train_gpu.py`

---

## 9) Dépannage

### ImportError / package manquant
- mettre à jour `environment.yml`
- reconstruire l’image

### Données introuvables
- vérifier que le volume monte bien `-v ${PWD}:/app`
- vérifier que `DATA_PATH` est correct

### Permissions
- exécuter Docker Desktop avec les droits nécessaires

---

Si tu veux une image “prod” ultra‑légère (sans conda) ou un multi‑stage build, je peux le faire.
