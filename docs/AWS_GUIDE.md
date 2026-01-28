# Guide AWS — Déploiement du projet (détails complets)

## 1) Objectif
Déployer le projet ML sur AWS pour :
- exécuter l’entraînement sur une machine distante
- stocker les données et modèles
- (optionnel) exposer une API d’inférence

Ce guide propose une approche **simple et réaliste** pour ce projet :
- **EC2** pour l’exécution
- **S3** pour stocker datasets et modèles
- **(optionnel) ECR + ECS** pour conteneuriser

---

## 2) Pré‑requis AWS
- Compte AWS + IAM user
- AWS CLI installée
- Clés d’accès configurées (`aws configure`)

---

## 3) Architecture recommandée (simple)

### Variante A — EC2 + S3 (la plus simple)
1. Stocker données brutes et modèles sur **S3**
2. Lancer une instance **EC2**
3. Cloner le projet et exécuter `run_training.py`
4. Sauvegarder les outputs dans S3

### Variante B — Docker + ECS (plus propre)
1. Construire une image Docker
2. Pousser vers **ECR**
3. Exécuter un task ECS (Fargate)
4. Logs CloudWatch + artefacts sur S3

---

## 4) Déploiement avec EC2 + S3 (pas à pas)

### 4.1 Créer un bucket S3
- Créer un bucket, ex: `ml-project-sprint-bucket`
- Upload des CSV bruts et dossiers data

### 4.2 Lancer une instance EC2
- OS : Ubuntu 22.04
- Type : t3.large (CPU) ou g4dn.xlarge (GPU)
- Ouvrir SSH (port 22)

### 4.3 Connexion SSH
```bash
ssh -i "clé.pem" ubuntu@<public-ip>
```

### 4.4 Installer les dépendances
```bash
sudo apt update
sudo apt install -y git
```

Option 1 : Conda (si tu veux reproduire `environment.yml`)
```bash
# Installer Miniconda
# Puis
conda env create -f environment.yml
conda activate ml-predictive-maintenance
```

Option 2 : Pip (requirements.txt)
```bash
pip install -r requirements.txt
```

### 4.5 Cloner le projet
```bash
git clone <url-repo>
cd "ML project sprint"
```

### 4.6 Télécharger les données S3
```bash
aws s3 sync s3://ml-project-sprint-bucket/data ./data
```

### 4.7 Lancer le pipeline
```bash
python src/data/extract.py
python src/data/clean_simple.py
python create_target.py
python prepare_features.py
python run_training.py
```

### 4.8 Sauvegarder les résultats
```bash
aws s3 sync ./src/models/models s3://ml-project-sprint-bucket/models
```

---

## 5) Déploiement Docker avec ECS (optionnel)

### 5.1 Créer un dépôt ECR
- Créer un repository ECR (ex: `ml-project-sprint`)

### 5.2 Build + Push
```bash
docker build -f docker/Dockerfile -t ml-project-sprint .

aws ecr get-login-password | docker login --username AWS --password-stdin <account>.dkr.ecr.<region>.amazonaws.com

docker tag ml-project-sprint:latest <account>.dkr.ecr.<region>.amazonaws.com/ml-project-sprint:latest

docker push <account>.dkr.ecr.<region>.amazonaws.com/ml-project-sprint:latest
```

### 5.3 Créer une task ECS
- Choisir Fargate
- Définir CPU/RAM
- Ajouter variables d’environnement (WANDB, etc.)
- Log vers CloudWatch

### 5.4 Lancer la task
La task exécutera `run_training.py` par défaut.

---

## 6) W&B sur AWS
- Définir `WANDB_API_KEY` sur l’instance EC2 ou dans ECS task definition.
- Optionnel : utiliser `WANDB_MODE=online`.

---

## 7) Coûts (à surveiller)
- EC2 : coût horaire selon instance
- S3 : stockage + transferts
- ECS/Fargate : CPU/RAM à la minute

---

## 8) Recommandations
- Commencer par EC2 + S3 (simple)
- Passer à ECS uniquement si besoin d’automatisation
- Sauvegarder tous les modèles dans S3
- Utiliser des tags W&B pour chaque run

---

Si tu veux une version **production** (API d’inférence + autoscaling + monitoring), je peux l’écrire aussi.
