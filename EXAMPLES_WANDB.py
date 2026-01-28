"""
Exemples d'intégration WandB pour différents scénarios.
Copiez-collez le code qui vous convient!
"""

# ============================================================
# EXEMPLE 1: Intégration basique dans un script d'entraînement
# ============================================================

def example_basic_training():
    """Entraînement simple avec WandB"""
    import wandb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    import pandas as pd
    
    # Initialiser WandB
    wandb.init(
        project="industrial-failure-prediction",
        config={
            "model": "random_forest",
            "n_estimators": 100,
            "max_depth": 20,
            "test_size": 0.2
        }
    )
    
    # Charger les données
    df = pd.read_csv("data/processed/cleaned_data/sensor_data_cleaned.csv")
    X = df.drop("failure_within_24h", axis=1)
    y = df["failure_within_24h"]
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Entraîner
    model = RandomForestClassifier(n_estimators=100, max_depth=20)
    model.fit(X_train, y_train)
    
    # Évaluer
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    # Enregistrer les métriques
    wandb.log({
        "accuracy": accuracy,
        "auc": auc,
        "train_size": len(X_train),
        "test_size": len(X_test)
    })
    
    wandb.finish()


# ============================================================
# EXEMPLE 2: Entraînement avec epochs (progression)
# ============================================================

def example_training_with_epochs():
    """Entraînement avec suivi par epoch"""
    import wandb
    from sklearn.ensemble import GradientBoostingClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    import pandas as pd
    
    wandb.init(project="industrial-failure-prediction")
    
    df = pd.read_csv("data/processed/cleaned_data/sensor_data_cleaned.csv")
    X = df.drop("failure_within_24h", axis=1)
    y = df["failure_within_24h"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Entraîner avec warmstart (apprentissage progressif)
    model = GradientBoostingClassifier(
        n_estimators=1,
        warm_start=True,
        random_state=42
    )
    
    # Entraîner progressivement
    for epoch in range(1, 11):
        model.n_estimators = epoch * 10
        model.fit(X_train, y_train)
        
        # Évaluer
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        # Enregistrer
        wandb.log({
            "epoch": epoch,
            "accuracy": accuracy,
            "auc": auc,
            "n_estimators": epoch * 10
        })
    
    wandb.finish()


# ============================================================
# EXEMPLE 3: Comparer plusieurs modèles
# ============================================================

def example_compare_models():
    """Entraîner et comparer plusieurs modèles"""
    import wandb
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score
    import pandas as pd
    
    df = pd.read_csv("data/processed/cleaned_data/sensor_data_cleaned.csv")
    X = df.drop("failure_within_24h", axis=1)
    y = df["failure_within_24h"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    models = {
        "random_forest": RandomForestClassifier(n_estimators=100),
        "gradient_boosting": GradientBoostingClassifier(n_estimators=100),
        "logistic_regression": LogisticRegression(max_iter=1000)
    }
    
    results = {}
    
    for model_name, model in models.items():
        # Créer un run par modèle
        wandb.init(
            project="industrial-failure-prediction",
            name=f"comparison-{model_name}",
            config={"model": model_name}
        )
        
        # Entraîner
        model.fit(X_train, y_train)
        
        # Évaluer
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_pred)
        
        # Enregistrer
        wandb.log({
            "accuracy": accuracy,
            "auc": auc
        })
        
        results[model_name] = {"accuracy": accuracy, "auc": auc}
        
        wandb.finish()
    
    # Afficher les résultats
    print("\n📊 Résultats de la comparaison:")
    for model_name, metrics in results.items():
        print(f"{model_name}: {metrics}")


# ============================================================
# EXEMPLE 4: Enregistrer des visualisations
# ============================================================

def example_log_visualizations():
    """Enregistrer des graphiques dans WandB"""
    import wandb
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, roc_curve
    import numpy as np
    
    wandb.init(project="industrial-failure-prediction")
    
    # Données de test (simulées)
    y_true = np.array([0, 1, 0, 1, 1, 0, 1, 0, 1, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 0, 1, 0, 1, 1])
    y_proba = np.array([0.1, 0.9, 0.2, 0.8, 0.4, 0.1, 0.85, 0.1, 0.95, 0.9])
    
    # === Matrice de confusion ===
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    im = ax.imshow(cm, cmap=plt.cm.Blues)
    ax.set_xlabel("Prédiction")
    ax.set_ylabel("Réalité")
    ax.set_title("Matrice de Confusion")
    plt.colorbar(im)
    
    wandb.log({"confusion_matrix": wandb.Image(fig)})
    plt.close()
    
    # === Courbe ROC ===
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label="ROC")
    ax.plot([0, 1], [0, 1], 'k--', label="Random")
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title("Courbe ROC")
    ax.legend()
    
    wandb.log({"roc_curve": wandb.Image(fig)})
    plt.close()
    
    # === Histogramme de distribution ===
    fig, ax = plt.subplots()
    ax.hist(y_proba[y_true==0], bins=10, alpha=0.5, label="Négatif")
    ax.hist(y_proba[y_true==1], bins=10, alpha=0.5, label="Positif")
    ax.set_xlabel("Probabilité")
    ax.set_ylabel("Fréquence")
    ax.set_title("Distribution des Prédictions")
    ax.legend()
    
    wandb.log({"prediction_distribution": wandb.Image(fig)})
    plt.close()
    
    wandb.finish()


# ============================================================
# EXEMPLE 5: Enregistrer les feature importance
# ============================================================

def example_log_feature_importance():
    """Enregistrer l'importance des features"""
    import wandb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    import pandas as pd
    
    wandb.init(project="industrial-failure-prediction")
    
    # Charger et entraîner
    df = pd.read_csv("data/processed/cleaned_data/sensor_data_cleaned.csv")
    X = df.drop("failure_within_24h", axis=1)
    y = df["failure_within_24h"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Obtenir l'importance
    importance_dict = dict(zip(X.columns, model.feature_importances_))
    
    # Créer un DataFrame
    importance_df = pd.DataFrame(
        list(importance_dict.items()),
        columns=["feature", "importance"]
    ).sort_values("importance", ascending=False)
    
    # Enregistrer comme table
    table = wandb.Table(dataframe=importance_df)
    wandb.log({"feature_importance": table})
    
    # Aussi enregistrer le top 10 comme graphique
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(10, 6))
    top_10 = importance_df.head(10)
    ax.barh(top_10["feature"], top_10["importance"])
    ax.set_xlabel("Importance")
    ax.set_title("Top 10 Features")
    
    wandb.log({"feature_importance_plot": wandb.Image(fig)})
    plt.close()
    
    wandb.finish()


# ============================================================
# EXEMPLE 6: Grouper les runs par expérience
# ============================================================

def example_group_runs():
    """Grouper plusieurs runs par expérience"""
    import wandb
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split, GridSearchCV
    import pandas as pd
    
    df = pd.read_csv("data/processed/cleaned_data/sensor_data_cleaned.csv")
    X = df.drop("failure_within_24h", axis=1)
    y = df["failure_within_24h"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Paramètres à tester
    param_grid = {
        "n_estimators": [50, 100, 200],
        "max_depth": [10, 20, 30]
    }
    
    # Créer un groupe pour cette expérience
    for n_est in param_grid["n_estimators"]:
        for max_d in param_grid["max_depth"]:
            # Chaque combinaison est un run dans le même groupe
            wandb.init(
                project="industrial-failure-prediction",
                group="hyperparameter-tuning",  # ← Groupe
                name=f"rf-n_est={n_est}-max_d={max_d}",
                config={
                    "n_estimators": n_est,
                    "max_depth": max_d
                }
            )
            
            # Entraîner
            model = RandomForestClassifier(
                n_estimators=n_est,
                max_depth=max_d
            )
            model.fit(X_train, y_train)
            
            # Évaluer
            accuracy = model.score(X_test, y_test)
            wandb.log({"accuracy": accuracy})
            
            wandb.finish()


# ============================================================
# EXEMPLE 7: Utiliser le WandBHelper
# ============================================================

def example_wandb_helper():
    """Utiliser la classe WandBHelper"""
    from wandb.wandb_helper import WandBHelper
    import pandas as pd
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
    
    # Créer une instance du helper
    helper = WandBHelper(
        project_name="industrial-failure-prediction",
        enable=True
    )
    
    # Initialiser
    helper.init(
        name="example-with-helper",
        config={"model": "random_forest", "n_estimators": 100},
        tags=["example"],
        notes="Démonstration du WandBHelper"
    )
    
    # Charger les données
    df = pd.read_csv("data/processed/cleaned_data/sensor_data_cleaned.csv")
    X = df.drop("failure_within_24h", axis=1)
    y = df["failure_within_24h"]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
    
    # Entraîner
    model = RandomForestClassifier(n_estimators=100)
    model.fit(X_train, y_train)
    
    # Évaluer
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred)
    
    # Enregistrer les métriques
    helper.log_metrics({
        "accuracy": accuracy,
        "auc": auc
    })
    
    # Enregistrer l'importance des features
    importance_dict = dict(zip(X.columns, model.feature_importances_))
    helper.log_feature_importance(importance_dict, "random_forest")
    
    # Enregistrer la matrice de confusion
    helper.log_confusion_matrix(y_test, y_pred)
    
    # Terminer
    helper.finish()


if __name__ == "__main__":
    # Décommenter l'exemple que vous voulez tester
    
    # example_basic_training()
    # example_training_with_epochs()
    # example_compare_models()
    # example_log_visualizations()
    # example_log_feature_importance()
    # example_group_runs()
    # example_wandb_helper()
    
    print("✅ Exemples disponibles!")
    print("Décommentez la fonction que vous voulez tester dans main()")
