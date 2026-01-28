import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Import W&B (optionnel)
try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
from sklearn.metrics import confusion_matrix, roc_curve, precision_recall_curve, auc
from sklearn.metrics import classification_report, average_precision_score
import joblib
import os


def load_test_data(test_data_path):
    """
    Charge les données de test prétraitées pour l'évaluation du modèle.
    
    Args:
        test_data_path: Chemin vers les données de test
        
    Returns:
        X_test: Features de test
        y_test: Labels de test
    """
    try:
        test_data = pd.read_csv(test_data_path)
        # Supposons que la dernière colonne est la cible
        X_test = test_data.iloc[:, :-1]
        y_test = test_data.iloc[:, -1]
        return X_test, y_test
    except Exception as e:
        print(f"Erreur lors du chargement des données de test: {e}")
        return None, None


def load_model(model_path):
    """
    Charge un modèle entraîné.
    
    Args:
        model_path: Chemin vers le modèle sauvegardé
        
    Returns:
        model: Modèle chargé
    """
    try:
        model = joblib.load(model_path)
        return model
    except Exception as e:
        print(f"Erreur lors du chargement du modèle: {e}")
        return None


def calculate_classification_metrics(y_true, y_pred, y_prob=None):
    """
    Calcule les métriques de classification standards.
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        y_prob: Probabilités de prédiction (optionnel)
        
    Returns:
        metrics: Dictionnaire contenant les métriques calculées
    """
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted'),
        'recall': recall_score(y_true, y_pred, average='weighted'),
        'f1': f1_score(y_true, y_pred, average='weighted'),
    }
    
    if y_prob is not None and len(np.unique(y_true)) == 2:
        # Cas binaire avec probabilités
        metrics['auc'] = auc(
            *roc_curve(y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob)[:2]
        )
        metrics['avg_precision'] = average_precision_score(
            y_true, y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        )
    
    return metrics


def plot_confusion_matrix(y_true, y_pred, class_names=None, save_path=None):
    """
    Trace et sauvegarde la matrice de confusion.
    
    Args:
        y_true: Labels réels
        y_pred: Labels prédits
        class_names: Noms des classes (optionnel)
        save_path: Chemin pour sauvegarder le graphique (optionnel)
        
    Returns:
        None
    """
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Prédictions')
    plt.ylabel('Valeurs réelles')
    plt.title('Matrice de confusion')
    
    if save_path:
        plt.savefig(save_path)
    
    # Log to W&B
    if WANDB_AVAILABLE:
        try:
            wandb.log({"confusion_matrix": wandb.Image(plt)})
        except Exception:
            pass
    
    plt.show()


def plot_roc_curve(y_true, y_prob, save_path=None):
    """
    Trace et sauvegarde la courbe ROC (pour les problèmes binaires).
    
    Args:
        y_true: Labels réels
        y_prob: Probabilités de prédiction
        save_path: Chemin pour sauvegarder le graphique (optionnel)
        
    Returns:
        None
    """
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Taux de faux positifs')
    plt.ylabel('Taux de vrais positifs')
    plt.title('Courbe ROC')
    plt.legend(loc="lower right")
    
    if save_path:
        plt.savefig(save_path)
    
    # Log to W&B
    if WANDB_AVAILABLE:
        try:
            wandb.log({"roc_curve": wandb.Image(plt), "roc_auc": roc_auc})
        except Exception:
            pass
    
    plt.show()


def plot_precision_recall_curve(y_true, y_prob, save_path=None):
    """
    Trace et sauvegarde la courbe Precision-Recall (pour les problèmes binaires).
    
    Args:
        y_true: Labels réels
        y_prob: Probabilités de prédiction
        save_path: Chemin pour sauvegarder le graphique (optionnel)
        
    Returns:
        None
    """
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_precision = average_precision_score(y_true, y_prob)
    
    plt.figure(figsize=(10, 8))
    plt.plot(recall, precision, color='blue', lw=2, 
             label=f'Precision-Recall curve (AP = {avg_precision:.2f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Courbe Precision-Recall')
    plt.legend(loc="lower left")
    
    if save_path:
        plt.savefig(save_path)
    
    # Log to W&B
    if WANDB_AVAILABLE:
        try:
            wandb.log({"pr_curve": wandb.Image(plt), "avg_precision": avg_precision})
        except Exception:
            pass
    
    plt.show()


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """
    Trace et sauvegarde l'importance des caractéristiques.
    
    Args:
        model: Modèle entraîné avec un attribut feature_importances_
        feature_names: Noms des caractéristiques
        top_n: Nombre de caractéristiques à afficher
        save_path: Chemin pour sauvegarder le graphique (optionnel)
        
    Returns:
        None
    """
    try:
        # Vérifier si le modèle a un attribut feature_importances_
        importances = model.feature_importances_
        
        # Créer un DataFrame pour faciliter le tri
        feature_imp = pd.DataFrame({'feature': feature_names, 'importance': importances})
        feature_imp = feature_imp.sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(12, 8))
        sns.barplot(x='importance', y='feature', data=feature_imp)
        plt.title(f'Top {top_n} caractéristiques les plus importantes')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path)
        
        # Log to W&B
        if WANDB_AVAILABLE:
            try:
                wandb.log({"feature_importance": wandb.Image(plt)})
            except Exception:
                pass
        
        plt.show()
        
    except (AttributeError, TypeError) as e:
        print(f"Ce modèle ne prend pas en charge l'affichage de l'importance des caractéristiques: {e}")


def calculate_economic_impact(y_test, y_pred, y_prob, cost_matrix):
    """
    Calcule l'impact économique d'un modèle de maintenance prédictive.
    Args:
        y_test: Labels réels
        y_pred: Prédictions binaires
        y_prob: Probabilités de prédiction
        cost_matrix: Dictionnaire des coûts (TP, FP, FN, TN)
    Returns:
        dict avec total_cost, baseline_cost, savings, cost_reduction_pct
    """
    tn = ((y_test == 0) & (y_pred == 0)).sum()
    fp = ((y_test == 0) & (y_pred == 1)).sum()
    fn = ((y_test == 1) & (y_pred == 0)).sum()
    tp = ((y_test == 1) & (y_pred == 1)).sum()
    total_cost = (
        tn * cost_matrix['true_negative'] +
        fp * cost_matrix['false_positive'] +
        fn * cost_matrix['false_negative'] +
        tp * cost_matrix['true_positive']
    )
    baseline_cost = (y_test.sum() * cost_matrix['false_negative'])
    savings = baseline_cost - total_cost
    cost_reduction_pct = (savings / baseline_cost) * 100 if baseline_cost > 0 else 0
    return {
        'total_cost': total_cost,
        'baseline_cost': baseline_cost,
        'savings': savings,
        'cost_reduction_pct': cost_reduction_pct
    }


def find_optimal_threshold(y_test, y_prob, cost_matrix):
    """
    Recherche le seuil optimal pour minimiser le coût total.
    Args:
        y_test: Labels réels
        y_prob: Probabilités de prédiction
        cost_matrix: Dictionnaire des coûts
    Returns:
        dict avec optimal_threshold, optimal_cost, confusion_matrix
    """
    thresholds = np.linspace(0.01, 0.99, 99)
    results = []
    for threshold in thresholds:
        y_pred_threshold = (y_prob >= threshold).astype(int)
        tn = ((y_test == 0) & (y_pred_threshold == 0)).sum()
        fp = ((y_test == 0) & (y_pred_threshold == 1)).sum()
        fn = ((y_test == 1) & (y_pred_threshold == 0)).sum()
        tp = ((y_test == 1) & (y_pred_threshold == 1)).sum()
        total_cost = (
            tn * cost_matrix['true_negative'] +
            fp * cost_matrix['false_positive'] +
            fn * cost_matrix['false_negative'] +
            tp * cost_matrix['true_positive']
        )
        results.append({
            'threshold': threshold,
            'total_cost': total_cost,
            'tp': tp,
            'fp': fp,
            'tn': tn,
            'fn': fn
        })
    optimal_result = min(results, key=lambda x: x['total_cost'])
    optimal_threshold = optimal_result['threshold']
    return {
        'optimal_threshold': optimal_threshold,
        'optimal_cost': optimal_result['total_cost'],
        'confusion_matrix': [optimal_result['tp'], optimal_result['fp'], optimal_result['fn'], optimal_result['tn']]
    }


def evaluate_model(model, X_test, y_test, class_names=None, output_dir=None, cost_matrix=None):
    """
    Fonction principale pour effectuer l'évaluation complète du modèle.
    
    Args:
        model: Modèle entraîné
        X_test: Features de test
        y_test: Labels de test
        class_names: Noms des classes (optionnel)
        output_dir: Répertoire de sortie pour les graphiques (optionnel)
        cost_matrix: Matrice de coûts pour l'évaluation économique (optionnel)
        
    Returns:
        results: Dictionnaire contenant les résultats de l'évaluation
    """
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Faire des prédictions
    y_pred = model.predict(X_test)
    
    # Obtenir les probabilités si disponibles
    if hasattr(model, "predict_proba"):
        y_prob = model.predict_proba(X_test)
    else:
        y_prob = None
    
    # Calculer les métriques de classification
    metrics = calculate_classification_metrics(y_test, y_pred, y_prob)
    
    # Log métriques W&B
    if WANDB_AVAILABLE:
        try:
            wandb.log({f"eval/{k}": v for k, v in metrics.items()})
        except Exception:
            pass
    
    # Afficher les résultats
    print("\n==== Résultats de l'évaluation ====")
    for metric, value in metrics.items():
        print(f"{metric}: {value:.4f}")
    
    # Rapport de classification détaillé
    print("\n==== Rapport de classification ====")
    print(classification_report(y_test, y_pred))
    
    # Tracer et sauvegarder la matrice de confusion
    cm_path = os.path.join(output_dir, "confusion_matrix.png") if output_dir else None
    plot_confusion_matrix(y_test, y_pred, class_names, cm_path)
    
    # Pour les problèmes binaires, tracer ROC et PR curves
    binary_classification = len(np.unique(y_test)) == 2
    if binary_classification and y_prob is not None:
        # Pour les problèmes binaires, obtenir les probabilités de la classe positive
        prob_positive = y_prob[:, 1] if y_prob.ndim > 1 else y_prob
        
        # Tracer la courbe ROC
        roc_path = os.path.join(output_dir, "roc_curve.png") if output_dir else None
        plot_roc_curve(y_test, prob_positive, roc_path)
        
        # Tracer la courbe Precision-Recall
        pr_path = os.path.join(output_dir, "pr_curve.png") if output_dir else None
        plot_precision_recall_curve(y_test, prob_positive, pr_path)
    
    # Tracer l'importance des caractéristiques si disponible
    if hasattr(model, "feature_importances_"):
        fi_path = os.path.join(output_dir, "feature_importance.png") if output_dir else None
        plot_feature_importance(model, X_test.columns, save_path=fi_path)
    
    # Évaluation économique si la matrice de coûts est fournie
    if cost_matrix is not None and isinstance(cost_matrix, dict):
        # Trouver le seuil optimal
        threshold_results = find_optimal_threshold(y_test, y_prob[:, 1], cost_matrix)
        optimal_threshold = threshold_results['optimal_threshold']
        print(f"\nSeuil optimal trouvé: {optimal_threshold:.4f}")
        
        # Prédictions avec le seuil optimal
        y_pred_optimal = (y_prob[:, 1] >= optimal_threshold).astype(int)
        
        # Calculer l'impact économique
        economic_impact = calculate_economic_impact(y_test, y_pred_optimal, y_prob, cost_matrix)
        print("\n==== Impact économique ====")
        for key, value in economic_impact.items():
            print(f"{key}: {value}")
        
        # Log impact économique W&B
        if WANDB_AVAILABLE:
            try:
                wandb.log({
                    "economic/optimal_threshold": optimal_threshold,
                    "economic/total_cost": economic_impact['total_cost'],
                    "economic/baseline_cost": economic_impact['baseline_cost'],
                    "economic/savings": economic_impact['savings'],
                    "economic/cost_reduction_pct": economic_impact['cost_reduction_pct']
                })
            except Exception:
                pass
        
        # Matrice de confusion avec le seuil optimal
        cm_optimal = confusion_matrix(y_test, y_pred_optimal)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm_optimal, annot=True, fmt='d', cmap='Greens', 
                    xticklabels=class_names, yticklabels=class_names)
        plt.xlabel('Prédictions (seuil optimal)')
        plt.ylabel('Valeurs réelles')
        plt.title('Matrice de confusion avec seuil optimal')
        
        # Log matrice optimale W&B
        if WANDB_AVAILABLE:
            try:
                wandb.log({"confusion_matrix_optimal": wandb.Image(plt)})
            except Exception:
                pass
        
        plt.show()
    
    return {
        'metrics': metrics,
        'y_true': y_test,
        'y_pred': y_pred,
        'y_prob': y_prob
    }


if __name__ == "__main__":
    # Exemple d'utilisation du script
    model_path = "models/predictive_maintenance_model.pkl"
    test_data_path = "data/processed/feature_test_data.csv"
    output_dir = "reports/evaluation"
    
    # Charger les données et le modèle
    X_test, y_test = load_test_data(test_data_path)
    model = load_model(model_path)
    
    if X_test is not None and model is not None:
        # Classes pour les défaillances (à adapter selon vos données)
        class_names = ["Pas de défaillance", "Défaillance"]
        
        # Matrice de coûts exemple (à adapter selon vos besoins)
        cost_matrix = {
            'true_positive': 10000,   # Gain pour une détection correcte
            'false_positive': -1000,  # Coût d'une fausse alerte
            'true_negative': 100,     # Gain pour une non-détection correcte
            'false_negative': -5000    # Coût d'une détection manquée
        }
        
        # Évaluer le modèle
        results = evaluate_model(model, X_test, y_test, class_names, output_dir, cost_matrix)
        print("\nÉvaluation du modèle terminée avec succès!")
    else:
        print("Impossible de procéder à l'évaluation du modèle.")
