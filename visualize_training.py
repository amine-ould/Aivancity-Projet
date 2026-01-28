#!/usr/bin/env python
"""
Visualisation et comparaison des performances des modèles
Génère des graphes: Accuracy, AUC, Confusion Matrix, ROC
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_curve, auc, accuracy_score, roc_auc_score
import joblib
from glob import glob

# Configuration
DATA_PATH = r"data\processed\cleaned_data\sensor_data_cleaned.csv"
MODELS_DIR = "src/models/models"
TARGET_COLUMN = "failure_within_24h"

print("\n" + "="*70)
print("📊 VISUALISATION DES RÉSULTATS D'ENTRAÎNEMENT")
print("="*70 + "\n")

# ===== CHARGER LES DONNÉES =====
print("📂 Chargement des données...")
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=[TARGET_COLUMN])
y = df[TARGET_COLUMN]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print(f"✅ Test set: {X_test.shape[0]} échantillons\n")

# ===== CHARGER LES MODÈLES =====
print("🤖 Chargement des modèles...")
model_files = glob(os.path.join(MODELS_DIR, "*.pkl"))

if not model_files:
    print(f"❌ Aucun modèle trouvé dans {MODELS_DIR}")
    exit(1)

models = {}
for file in model_files:
    name = os.path.basename(file).replace(".pkl", "").split("_")[0]
    models[name] = joblib.load(file)
    print(f"   ✅ {name}")

# ===== ÉVALUATION =====
print("\n🔍 Évaluation des modèles...")
results = {}

for name, model in models.items():
    try:
        y_pred = model.predict(X_test)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        
        accuracy = accuracy_score(y_test, y_pred)
        auc_score = roc_auc_score(y_test, y_pred_proba)
        
        results[name] = {
            'accuracy': accuracy,
            'auc': auc_score,
            'y_pred': y_pred,
            'y_pred_proba': y_pred_proba
        }
        print(f"   {name:20} → Accuracy: {accuracy:.4f} | AUC: {auc_score:.4f}")
    except Exception as e:
        print(f"   ❌ {name}: {str(e)[:50]}")

# ===== GRAPHIQUES =====
print("\n📈 Génération des graphes...")

# 1. Comparaison Accuracy vs AUC
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

names = list(results.keys())
accuracies = [results[n]['accuracy'] for n in names]
aucs = [results[n]['auc'] for n in names]

# Graphique Accuracy
axes[0].bar(names, accuracies, color='skyblue', edgecolor='black')
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].set_title('Comparaison des Accuracy', fontsize=14, fontweight='bold')
axes[0].set_ylim([0.95, 1.0])
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(accuracies):
    axes[0].text(i, v+0.001, f'{v:.4f}', ha='center', fontsize=10)
axes[0].tick_params(axis='x', rotation=45)

# Graphique AUC
axes[1].bar(names, aucs, color='lightgreen', edgecolor='black')
axes[1].set_ylabel('AUC-ROC', fontsize=12)
axes[1].set_title('Comparaison des AUC-ROC', fontsize=14, fontweight='bold')
axes[1].set_ylim([0, 1])
axes[1].grid(axis='y', alpha=0.3)
for i, v in enumerate(aucs):
    axes[1].text(i, v+0.02, f'{v:.4f}', ha='center', fontsize=10)
axes[1].tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.savefig('results_accuracy_auc.png', dpi=300, bbox_inches='tight')
print("   ✅ Sauvegardé: results_accuracy_auc.png")
plt.close()

# 2. Confusion Matrix pour chaque modèle
n_models = len(results)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, (name, res) in enumerate(results.items()):
    cm = confusion_matrix(y_test, res['y_pred'])
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False)
    axes[idx].set_title(f'{name}\nAccuracy: {res["accuracy"]:.4f}', fontweight='bold')
    axes[idx].set_ylabel('Réelle')
    axes[idx].set_xlabel('Prédite')
    axes[idx].set_xticklabels(['Pas Panne', 'Panne'])
    axes[idx].set_yticklabels(['Pas Panne', 'Panne'])

# Masquer les axes inutilisés
for idx in range(len(results), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Matrices de Confusion - Tous les Modèles', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('results_confusion_matrices.png', dpi=300, bbox_inches='tight')
print("   ✅ Sauvegardé: results_confusion_matrices.png")
plt.close()

# 3. Courbes ROC
fig, ax = plt.subplots(figsize=(10, 8))

colors = plt.cm.Set1(np.linspace(0, 1, len(results)))

for (name, res), color in zip(results.items(), colors):
    fpr, tpr, _ = roc_curve(y_test, res['y_pred_proba'])
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, label=f'{name} (AUC={roc_auc:.4f})', linewidth=2.5, color=color)

# Diagonal (modèle aléatoire)
ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Aléatoire (AUC=0.5)')

ax.set_xlabel('Taux de Faux Positifs (FPR)', fontsize=12)
ax.set_ylabel('Taux de Vrais Positifs (TPR)', fontsize=12)
ax.set_title('Courbes ROC - Comparaison des Modèles', fontsize=14, fontweight='bold')
ax.legend(loc='lower right', fontsize=11)
ax.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('results_roc_curves.png', dpi=300, bbox_inches='tight')
print("   ✅ Sauvegardé: results_roc_curves.png")
plt.close()

# 4. Tableau récapitulatif
print("\n" + "="*70)
print("📊 RÉSUMÉ DES PERFORMANCES")
print("="*70 + "\n")

results_df = pd.DataFrame({
    'Modèle': names,
    'Accuracy': [f"{results[n]['accuracy']:.4f}" for n in names],
    'AUC-ROC': [f"{results[n]['auc']:.4f}" for n in names]
})

print(results_df.to_string(index=False))
print()

# Meilleur modèle
best_model = max(results.items(), key=lambda x: x[1]['auc'])
print(f"🏆 Meilleur modèle: {best_model[0]} (AUC={best_model[1]['auc']:.4f})")

print("\n" + "="*70)
print("✅ Graphes générés:")
print("   📈 results_accuracy_auc.png")
print("   📊 results_confusion_matrices.png")
print("   🎯 results_roc_curves.png")
print("="*70 + "\n")
