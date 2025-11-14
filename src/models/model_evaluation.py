# ===============================================================
# 📌 EVALUACIÓN COMPLETA DEL MODELO
# ===============================================================

import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from mlflow.tracking import MlflowClient
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, log_loss, confusion_matrix,
    roc_curve, precision_recall_curve, classification_report
)
from pathlib import Path
import os


EXPERIMENT_NAME = "churn-prediction-experiment"
MODEL_NAME = "churn-prediction-model"


def load_best_model(model_name=MODEL_NAME, stage="Production"):
    """Cargar el mejor modelo desde MLflow Model Registry"""
    client = MlflowClient()
    
    try:
        model_version = client.get_latest_versions(model_name, stages=[stage])[0]
        model_uri = f"models:/{model_name}/{stage}"
        
        # Intentar cargar con diferentes backends
        model = None
        for loader in [mlflow.sklearn.load_model, mlflow.xgboost.load_model, mlflow.lightgbm.load_model]:
            try:
                model = loader(model_uri)
                break
            except:
                continue
        
        if model is None:
            raise Exception("No se pudo cargar el modelo con ningún backend disponible")
        
        print(f"✅ Modelo cargado: {model_name} v{model_version.version}")
        return model, model_version
    except Exception as e:
        print(f"⚠️ Error cargando modelo desde Registry: {e}")
        print("📥 Intentando cargar desde último run...")
        
        # Cargar desde último run del experimento
        experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
        runs = client.search_runs(
            experiment.experiment_id,
            order_by=["metrics.roc_auc DESC"],
            max_results=1
        )
        
        if runs:
            best_run = runs[0]
            model_uri = f"runs:/{best_run.info.run_id}/model"
            
            # Intentar cargar con diferentes backends
            model = None
            for loader in [mlflow.sklearn.load_model, mlflow.xgboost.load_model, mlflow.lightgbm.load_model]:
                try:
                    model = loader(model_uri)
                    break
                except:
                    continue
            
            if model is None:
                raise Exception("No se pudo cargar el modelo con ningún backend disponible")
            
            print(f"✅ Modelo cargado desde run: {best_run.info.run_id}")
            return model, best_run
        else:
            raise Exception("No se encontró ningún modelo")


def calculate_metrics(y_true, y_pred, y_pred_proba):
    """Calcular todas las métricas de clasificación"""
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred),
        'recall': recall_score(y_true, y_pred),
        'f1_score': f1_score(y_true, y_pred),
        'roc_auc': roc_auc_score(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba)
    }
    return metrics


def plot_confusion_matrix(y_true, y_pred, save_path=None):
    """Visualizar matriz de confusión"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['No Abandono', 'Abandono'],
                yticklabels=['No Abandono', 'Abandono'])
    plt.title('Matriz de Confusión')
    plt.ylabel('Verdadero')
    plt.xlabel('Predicho')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Matriz de confusión guardada: {save_path}")
    
    plt.show()
    return cm


def plot_roc_curve(y_true, y_pred_proba, save_path=None):
    """Visualizar curva ROC"""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
    auc_score = roc_auc_score(y_true, y_pred_proba)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, 
             label=f'ROC curve (AUC = {auc_score:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', 
             label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos (FPR)')
    plt.ylabel('Tasa de Verdaderos Positivos (TPR)')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Curva ROC guardada: {save_path}")
    
    plt.show()
    return fpr, tpr, auc_score


def plot_precision_recall_curve(y_true, y_pred_proba, save_path=None):
    """Visualizar curva Precision-Recall"""
    precision, recall, thresholds = precision_recall_curve(y_true, y_pred_proba)
    avg_precision = np.trapz(precision, recall)
    
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, color='darkorange', lw=2,
             label=f'Precision-Recall (AP = {avg_precision:.4f})')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc="lower left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"✅ Curva Precision-Recall guardada: {save_path}")
    
    plt.show()
    return precision, recall, avg_precision


def plot_feature_importance(model, feature_names, top_n=20, save_path=None):
    """Visualizar feature importance"""
    try:
        # Intentar obtener feature importance
        if hasattr(model, 'feature_importances_'):
            importances = model.feature_importances_
        elif hasattr(model, 'coef_'):
            importances = np.abs(model.coef_[0])
        else:
            print("⚠️ Modelo no tiene feature importance disponible")
            return None
        
        # Crear DataFrame
        feature_imp_df = pd.DataFrame({
            'feature': feature_names[:len(importances)],
            'importance': importances
        }).sort_values('importance', ascending=False).head(top_n)
        
        # Visualizar
        plt.figure(figsize=(10, 8))
        plt.barh(range(len(feature_imp_df)), feature_imp_df['importance'])
        plt.yticks(range(len(feature_imp_df)), feature_imp_df['feature'])
        plt.xlabel('Importancia')
        plt.title(f'Top {top_n} Features más Importantes')
        plt.gca().invert_yaxis()
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✅ Feature importance guardada: {save_path}")
        
        plt.show()
        return feature_imp_df
    except Exception as e:
        print(f"⚠️ Error al calcular feature importance: {e}")
        return None


def evaluate_model(model, X_test, y_test, feature_names=None, save_dir=None):
    """Evaluación completa del modelo"""
    print("\n" + "="*70)
    print("📊 EVALUACIÓN COMPLETA DEL MODELO")
    print("="*70)
    
    # Crear directorio para guardar gráficos
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Predicciones
    print("\n🔮 Generando predicciones...")
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Métricas
    print("\n📈 Calculando métricas...")
    metrics = calculate_metrics(y_test, y_pred, y_pred_proba)
    
    print("\n" + "-"*70)
    print("MÉTRICAS DE CLASIFICACIÓN:")
    print("-"*70)
    for metric_name, metric_value in metrics.items():
        print(f"  {metric_name:15s}: {metric_value:.4f}")
    
    # Classification Report
    print("\n" + "-"*70)
    print("REPORTE DE CLASIFICACIÓN:")
    print("-"*70)
    print(classification_report(y_test, y_pred, 
                                target_names=['No Abandono', 'Abandono']))
    
    # Matriz de confusión
    print("\n📊 Generando matriz de confusión...")
    cm_path = save_dir / "confusion_matrix.png" if save_dir else None
    cm = plot_confusion_matrix(y_test, y_pred, cm_path)
    
    # Curva ROC
    print("\n📊 Generando curva ROC...")
    roc_path = save_dir / "roc_curve.png" if save_dir else None
    fpr, tpr, auc = plot_roc_curve(y_test, y_pred_proba, roc_path)
    
    # Curva Precision-Recall
    print("\n📊 Generando curva Precision-Recall...")
    pr_path = save_dir / "precision_recall_curve.png" if save_dir else None
    precision, recall, ap = plot_precision_recall_curve(y_test, y_pred_proba, pr_path)
    
    # Feature Importance
    if feature_names:
        print("\n📊 Generando feature importance...")
        fi_path = save_dir / "feature_importance.png" if save_dir else None
        feature_imp_df = plot_feature_importance(model, feature_names, top_n=20, save_path=fi_path)
    else:
        feature_imp_df = None
    
    # Resumen
    print("\n" + "="*70)
    print("✅ EVALUACIÓN COMPLETA FINALIZADA")
    print("="*70)
    
    return {
        'metrics': metrics,
        'confusion_matrix': cm,
        'roc_curve': (fpr, tpr, auc),
        'precision_recall_curve': (precision, recall, ap),
        'feature_importance': feature_imp_df,
        'predictions': y_pred,
        'prediction_probas': y_pred_proba
    }


def evaluate_on_all_splits(model, data_dict, feature_names=None, save_dir=None):
    """Evaluar modelo en train, val y test"""
    results = {}
    
    # Test set
    print("\n" + "="*70)
    print("🧪 EVALUACIÓN EN TEST SET")
    print("="*70)
    results['test'] = evaluate_model(
        model, 
        data_dict['X_test'], 
        data_dict['y_test'],
        feature_names,
        save_dir / "test" if save_dir else None
    )
    
    # Validation set
    print("\n" + "="*70)
    print("🔍 EVALUACIÓN EN VALIDATION SET")
    print("="*70)
    results['val'] = evaluate_model(
        model,
        data_dict['X_val'],
        data_dict['y_val'],
        feature_names,
        save_dir / "val" if save_dir else None
    )
    
    # Comparación
    print("\n" + "="*70)
    print("📊 COMPARACIÓN VAL vs TEST")
    print("="*70)
    comparison = pd.DataFrame({
        'Validation': results['val']['metrics'],
        'Test': results['test']['metrics']
    })
    print(comparison.to_string())
    
    return results


if __name__ == "__main__":
    from src.data.data_preparation import prepare_data
    
    print("🚀 Iniciando evaluación del modelo...")
    
    # Preparar datos
    data_dict = prepare_data()
    
    # Cargar mejor modelo
    model, model_info = load_best_model()
    
    # Evaluar
    save_dir = Path("results/evaluation")
    results = evaluate_on_all_splits(
        model, 
        data_dict,
        feature_names=data_dict.get('feature_names'),
        save_dir=save_dir
    )
    
    print("\n✅ Evaluación completada!")

