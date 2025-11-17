# ===============================================================
# 📌 MODELADO CON MLFLOW
# ===============================================================

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, roc_auc_score, log_loss
)
import numpy as np
import pandas as pd
from pathlib import Path
import os


# Configurar MLflow
MLFLOW_TRACKING_URI = "http://localhost:5000"
EXPERIMENT_NAME = "churn-prediction-experiment"


def setup_mlflow():
    """Configurar MLflow tracking"""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    print(f"✅ MLflow configurado: {MLFLOW_TRACKING_URI}")
    print(f"📊 Experimento: {EXPERIMENT_NAME}")


def train_logistic_regression(X_train, y_train, X_val, y_val):
    """Entrenar Logistic Regression con hyperparameter tuning"""
    print("\n" + "="*60)
    print("🔵 Entrenando Logistic Regression...")
    print("="*60)
    
    # Espacio de hiperparámetros
    param_grid = {
        'C': [0.001, 0.01, 0.1, 1, 10, 100],
        'penalty': ['l1', 'l2'],
        'solver': ['liblinear', 'saga'],
        'max_iter': [1000, 2000]
    }
    
    # Modelo base
    base_model = LogisticRegression(random_state=42)
    
    # Randomized Search
    random_search = RandomizedSearchCV(
        base_model, param_grid, 
        n_iter=20, cv=3, scoring='roc_auc',
        n_jobs=-1, random_state=42, verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Predicciones
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    
    # Métricas
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba),
        'log_loss': log_loss(y_val, y_pred_proba)
    }
    
    # Logging en MLflow
    with mlflow.start_run(run_name="LogisticRegression"):
        # Log parámetros
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "LogisticRegression")
        # NEW: Log full param search configuration
        mlflow.log_dict(param_grid, "hyperparameter_space_logistic_regression.json")

        # NEW: Log CV results (scikit returns a dict)
        cv_results = pd.DataFrame(random_search.cv_results_)
        mlflow.log_table(cv_results, artifact_file="cv_results_logistic_regression.json")
        # NEW: Log the SearchCV object for reproducibility
        mlflow.sklearn.log_model(random_search, "search_model_logistic_regression")
        
        # Log métricas
        mlflow.log_metrics(metrics)
        
        # Log modelo
        mlflow.sklearn.log_model(best_model, "model")
        
        # Log artefactos adicionales
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_samples_val", X_val.shape[0])
    
    print(f"✅ Mejores parámetros: {best_params}")
    print(f"📊 ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"📊 F1-Score: {metrics['f1_score']:.4f}")
    
    return best_model, metrics


def train_random_forest(X_train, y_train, X_val, y_val):
    """Entrenar Random Forest con hyperparameter tuning"""
    print("\n" + "="*60)
    print("🟢 Entrenando Random Forest...")
    print("="*60)
    
    # Espacio de hiperparámetros
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2', None]
    }
    
    # Modelo base
    base_model = RandomForestClassifier(random_state=42, n_jobs=-1)
    
    # Randomized Search
    random_search = RandomizedSearchCV(
        base_model, param_grid,
        n_iter=30, cv=3, scoring='roc_auc',
        n_jobs=-1, random_state=42, verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Predicciones
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    
    # Métricas
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba),
        'log_loss': log_loss(y_val, y_pred_proba)
    }
    
    # Logging en MLflow
    with mlflow.start_run(run_name="RandomForest"):
        # Log parámetros
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "RandomForest")
        # NEW: Log full param search configuration
        mlflow.log_dict(param_grid, "hyperparameter_space_random_forest.json")

        # NEW: Log CV results (scikit returns a dict)
        cv_results = pd.DataFrame(random_search.cv_results_)
        mlflow.log_table(cv_results, artifact_file="cv_results_random_forest.json")
        # NEW: Log the SearchCV object for reproducibility
        mlflow.sklearn.log_model(random_search, "search_model_random_forest")
        
        # Log métricas
        mlflow.log_metrics(metrics)
        
        # Log modelo
        mlflow.sklearn.log_model(best_model, "model")
        
        # Log artefactos adicionales
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_samples_val", X_val.shape[0])
    
    print(f"✅ Mejores parámetros: {best_params}")
    print(f"📊 ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"📊 F1-Score: {metrics['f1_score']:.4f}")
    
    return best_model, metrics


def train_xgboost(X_train, y_train, X_val, y_val):
    """Entrenar XGBoost con hyperparameter tuning"""
    print("\n" + "="*60)
    print("🟡 Entrenando XGBoost...")
    print("="*60)
    
    # Espacio de hiperparámetros
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 9],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'gamma': [0, 0.1, 0.2]
    }
    
    # Modelo base
    base_model = XGBClassifier(random_state=42, eval_metric='logloss')
    
    # Randomized Search
    random_search = RandomizedSearchCV(
        base_model, param_grid,
        n_iter=30, cv=3, scoring='roc_auc',
        n_jobs=-1, random_state=42, verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Predicciones
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    
    # Métricas
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba),
        'log_loss': log_loss(y_val, y_pred_proba)
    }
    
    # Logging en MLflow
    with mlflow.start_run(run_name="XGBoost"):
        # Log parámetros
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "XGBoost")
        
        # NEW: Log full param search configuration
        mlflow.log_dict(param_grid, "hyperparameter_space_xgboost.json")

        # NEW: Log CV results (scikit returns a dict)
        cv_results = pd.DataFrame(random_search.cv_results_)
        mlflow.log_table(cv_results, artifact_file="cv_results_xgboost.json")

        # NEW: Log the SearchCV object for reproducibility
        mlflow.sklearn.log_model(random_search, "search_model_xgboost")
        
        # Log métricas
        mlflow.log_metrics(metrics)
        
        # Log modelo
        mlflow.xgboost.log_model(best_model, "model")
        
        # Log artefactos adicionales
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_samples_val", X_val.shape[0])
    
    print(f"✅ Mejores parámetros: {best_params}")
    print(f"📊 ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"📊 F1-Score: {metrics['f1_score']:.4f}")
    
    return best_model, metrics


def train_lightgbm(X_train, y_train, X_val, y_val):
    """Entrenar LightGBM con hyperparameter tuning"""
    print("\n" + "="*60)
    print("🟣 Entrenando LightGBM...")
    print("="*60)
    
    # Espacio de hiperparámetros
    param_grid = {
        'n_estimators': [50, 100, 200, 300],
        'max_depth': [3, 5, 7, 9, -1],
        'learning_rate': [0.01, 0.1, 0.2, 0.3],
        'subsample': [0.6, 0.8, 1.0],
        'colsample_bytree': [0.6, 0.8, 1.0],
        'num_leaves': [31, 50, 100, 200],
        'reg_alpha': [0, 0.1, 0.5],
        'reg_lambda': [0, 0.1, 0.5]
    }
    
    # Modelo base
    base_model = LGBMClassifier(random_state=42, verbose=-1)
    
    # Randomized Search
    random_search = RandomizedSearchCV(
        base_model, param_grid,
        n_iter=30, cv=3, scoring='roc_auc',
        n_jobs=-1, random_state=42, verbose=1
    )
    
    random_search.fit(X_train, y_train)
    
    best_model = random_search.best_estimator_
    best_params = random_search.best_params_
    
    # Predicciones
    y_pred = best_model.predict(X_val)
    y_pred_proba = best_model.predict_proba(X_val)[:, 1]
    
    # Métricas
    metrics = {
        'accuracy': accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred),
        'recall': recall_score(y_val, y_pred),
        'f1_score': f1_score(y_val, y_pred),
        'roc_auc': roc_auc_score(y_val, y_pred_proba),
        'log_loss': log_loss(y_val, y_pred_proba)
    }
    
    # Logging en MLflow
    with mlflow.start_run(run_name="LightGBM"):
        # Log parámetros
        mlflow.log_params(best_params)
        mlflow.log_param("model_type", "LightGBM")

        # NEW: Log full param search configuration
        mlflow.log_dict(param_grid, "hyperparameter_space_lightgbm.json")

        # NEW: Log CV results (scikit returns a dict)
        cv_results = pd.DataFrame(random_search.cv_results_)
        mlflow.log_table(cv_results, artifact_file="cv_results_lightgbm.json")

        # NEW: Log the SearchCV object for reproducibility
        mlflow.sklearn.log_model(random_search, "search_model_lightgbm")

        # Log métricas
        mlflow.log_metrics(metrics)
        
        # Log modelo
        mlflow.lightgbm.log_model(best_model, "model")
        
        # Log artefactos adicionales
        mlflow.log_param("n_features", X_train.shape[1])
        mlflow.log_param("n_samples_train", X_train.shape[0])
        mlflow.log_param("n_samples_val", X_val.shape[0])
    
    print(f"✅ Mejores parámetros: {best_params}")
    print(f"📊 ROC-AUC: {metrics['roc_auc']:.4f}")
    print(f"📊 F1-Score: {metrics['f1_score']:.4f}")
    
    return best_model, metrics


def compare_models(mlflow_client, experiment_name):
    """Comparar todos los modelos y seleccionar el mejor"""
    print("\n" + "="*60)
    print("📊 COMPARACIÓN DE MODELOS")
    print("="*60)
    
    experiment = mlflow_client.get_experiment_by_name(experiment_name)
    runs = mlflow_client.search_runs(
        experiment.experiment_id,
        order_by=["metrics.roc_auc DESC"]
    )
    
    results = []
    for run in runs:
        results.append({
            'run_id': run.info.run_id,
            'model_type': run.data.params.get('model_type', 'Unknown'),
            'roc_auc': run.data.metrics.get('roc_auc', 0),
            'f1_score': run.data.metrics.get('f1_score', 0),
            'accuracy': run.data.metrics.get('accuracy', 0),
            'precision': run.data.metrics.get('precision', 0),
            'recall': run.data.metrics.get('recall', 0)
        })
    
    results_df = pd.DataFrame(results)
    print("\n📋 Resultados de todos los modelos:")
    print(results_df.to_string(index=False))
    
    # Mejor modelo
    best_run = runs[0]
    best_model_type = best_run.data.params.get('model_type', 'Unknown')
    best_roc_auc = best_run.data.metrics.get('roc_auc', 0)
    
    print(f"\n🏆 MEJOR MODELO:")
    print(f"   Tipo: {best_model_type}")
    print(f"   ROC-AUC: {best_roc_auc:.4f}")
    print(f"   Run ID: {best_run.info.run_id}")
    
    return best_run, results_df


def register_best_model(mlflow_client, best_run, model_name="churn-prediction-model"):
    """Registrar el mejor modelo en MLflow Model Registry"""
    print("\n" + "="*60)
    print("📦 REGISTRANDO MODELO EN MODEL REGISTRY")
    print("="*60)
    
    try:
        # Registrar modelo
        model_uri = f"runs:/{best_run.info.run_id}/model"
        mv = mlflow.register_model(model_uri, model_name)
        
        print(f"✅ Modelo registrado exitosamente!")
        print(f"   Nombre: {model_name}")
        print(f"   Versión: {mv.version}")
        print(f"   Stage: {mv.current_stage}")
        
        # Transicionar a Production
        mlflow_client.transition_model_version_stage(
            name=model_name,
            version=mv.version,
            stage="Production"
        )
        
        print(f"✅ Modelo promovido a Production")
        
        return mv
    except Exception as e:
        print(f"⚠️ Error al registrar modelo: {e}")
        return None


def train_all_models(data_dict):
    """Entrenar todos los modelos"""
    setup_mlflow()
    
    X_train = data_dict["X_train"]
    y_train = data_dict["y_train"]
    X_val = data_dict["X_val"]
    y_val = data_dict["y_val"]
    
    # Entrenar todos los modelos
    models = {}
    metrics_all = {}
    
    # Logistic Regression
    models['logistic'], metrics_all['logistic'] = train_logistic_regression(
        X_train, y_train, X_val, y_val
    )
    
    # Random Forest
    models['random_forest'], metrics_all['random_forest'] = train_random_forest(
        X_train, y_train, X_val, y_val
    )
    
    # XGBoost
    models['xgboost'], metrics_all['xgboost'] = train_xgboost(
        X_train, y_train, X_val, y_val
    )
    
    # LightGBM
    models['lightgbm'], metrics_all['lightgbm'] = train_lightgbm(
        X_train, y_train, X_val, y_val
    )
    
    # Comparar y seleccionar mejor modelo
    mlflow_client = mlflow.tracking.MlflowClient()
    best_run, comparison_df = compare_models(mlflow_client, EXPERIMENT_NAME)
    
    # Registrar mejor modelo
    registered_model = register_best_model(mlflow_client, best_run)
    
    return {
        'models': models,
        'metrics': metrics_all,
        'best_run': best_run,
        'comparison': comparison_df,
        'registered_model': registered_model
    }


if __name__ == "__main__":
    from src.data.data_preparation import prepare_data
    
    print("🚀 Iniciando entrenamiento de modelos...")
    
    # Preparar datos
    data_dict = prepare_data()
    
    # Entrenar modelos
    results = train_all_models(data_dict)
    
    print("\n✅ Entrenamiento completado!")
    print(f"📊 Mejor modelo: {results['best_run'].data.params.get('model_type')}")
    print(f"📊 Mejor ROC-AUC: {results['best_run'].data.metrics.get('roc_auc'):.4f}")

