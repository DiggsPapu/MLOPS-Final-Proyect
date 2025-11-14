# ===============================================================
# 📌 SCRIPT PRINCIPAL - PIPELINE COMPLETO
# ===============================================================

"""
Script principal para ejecutar el pipeline completo de MLOps:
1. Preparación de datos
2. Entrenamiento de modelos con MLflow
3. Evaluación completa
"""

import sys
from pathlib import Path

# Agregar src al path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.data.data_preparation import prepare_data
from src.models.model_training import train_all_models
from src.models.model_evaluation import load_best_model, evaluate_on_all_splits


def main():
    """Ejecutar pipeline completo"""
    print("="*70)
    print("🚀 PIPELINE COMPLETO DE MLOPS - PREDICCIÓN DE ABANDONO")
    print("="*70)
    
    # Paso 1: Preparación de datos
    print("\n" + "="*70)
    print("📊 PASO 1: PREPARACIÓN DE DATOS")
    print("="*70)
    data_dict = prepare_data()
    
    # Paso 2: Entrenamiento de modelos
    print("\n" + "="*70)
    print("🤖 PASO 2: ENTRENAMIENTO DE MODELOS CON MLFLOW")
    print("="*70)
    training_results = train_all_models(data_dict)
    
    # Paso 3: Evaluación completa
    print("\n" + "="*70)
    print("📈 PASO 3: EVALUACIÓN COMPLETA DEL MODELO")
    print("="*70)
    
    # Cargar mejor modelo
    model, model_info = load_best_model()
    
    # Evaluar en todos los splits
    save_dir = project_root / "results" / "evaluation"
    evaluation_results = evaluate_on_all_splits(
        model,
        data_dict,
        feature_names=data_dict.get('feature_names'),
        save_dir=save_dir
    )
    
    # Resumen final
    print("\n" + "="*70)
    print("✅ PIPELINE COMPLETADO EXITOSAMENTE")
    print("="*70)
    print(f"\n📊 Mejor modelo: {training_results['best_run'].data.params.get('model_type')}")
    print(f"📊 ROC-AUC (Val): {training_results['best_run'].data.metrics.get('roc_auc'):.4f}")
    print(f"📊 ROC-AUC (Test): {evaluation_results['test']['metrics']['roc_auc']:.4f}")
    print(f"\n📁 Resultados guardados en: {save_dir}")
    print(f"🌐 MLflow UI: http://localhost:5000")


if __name__ == "__main__":
    main()

