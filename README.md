# Proyecto Final - MLOps con CRISP-DM

## Predicción de Abandono de Clientes (Churn Prediction)

Proyecto completo de Machine Learning Operations siguiendo la metodología CRISP-DM para predecir el abandono de clientes en un call center.

## 📋 Estructura del Proyecto

```
MLOPS-Final-Proyect/
├── data/
│   └── synthetic/
│       └── synthetic_calls.csv          # Dataset sintético
├── notebooks/
│   └── EDA_Análisis_Exploratorio.ipynb  # Análisis exploratorio
├── src/
│   ├── data/
│   │   ├── generate_synthetic.py        # Generación de datos sintéticos
│   │   ├── data_preparation.py          # Pipeline de preparación de datos
│   │   └── pipeline.py                  # Pipeline original (legacy)
│   ├── models/
│   │   ├── model_training.py            # Entrenamiento con MLflow
│   │   └── model_evaluation.py          # Evaluación completa
│   └── main.py                          # Script principal
├── results/
│   └── evaluation/                      # Resultados de evaluación
├── requirements.txt                     # Dependencias
└── README.md                            # Este archivo
```

## 🚀 Instalación

### 1. Crear entorno virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
```

### 2. Instalar dependencias

```bash
pip install -r requirements.txt
```

### 3. Generar datos sintéticos (si no existen)

```bash
python src/data/generate_synthetic.py
```

## 📊 Uso

### Opción 1: Ejecutar pipeline completo

```bash
python src/main.py
```

Este script ejecuta:
1. Preparación de datos
2. Entrenamiento de 4 modelos con MLflow
3. Evaluación completa del mejor modelo

### Opción 2: Ejecutar componentes individuales

#### Preparar datos
```python
from src.data.data_preparation import prepare_data
data_dict = prepare_data()
```

#### Entrenar modelos
```python
from src.models.model_training import train_all_models
results = train_all_models(data_dict)
```

#### Evaluar modelo
```python
from src.models.model_evaluation import load_best_model, evaluate_on_all_splits
model, _ = load_best_model()
results = evaluate_on_all_splits(model, data_dict)
```

## 🔧 Configuración de MLflow

### 1. Iniciar servidor MLflow

En una terminal separada:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns
```

O si prefieres usar el servidor en localhost:5000:

```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

### 2. Acceder a la UI

Abre tu navegador en: `http://localhost:5000`

## 📈 Modelos Implementados

1. **Logistic Regression** - Modelo lineal básico
2. **Random Forest** - Ensemble de árboles de decisión
3. **XGBoost** - Gradient boosting optimizado
4. **LightGBM** - Gradient boosting rápido y eficiente

Todos los modelos incluyen:
- Hyperparameter tuning con RandomizedSearchCV
- Logging completo en MLflow
- Comparación sistemática
- Registro en Model Registry

## 📊 Métricas de Evaluación

- **Accuracy** - Precisión general
- **Precision** - Precisión de predicciones positivas
- **Recall** - Sensibilidad
- **F1-Score** - Media armónica de precision y recall
- **ROC-AUC** - Área bajo la curva ROC
- **Log Loss** - Pérdida logarítmica

## 📁 Resultados

Los resultados de la evaluación se guardan en `results/evaluation/`:
- Matriz de confusión
- Curvas ROC
- Curvas Precision-Recall
- Feature importance

## 🔍 Análisis Exploratorio

Ejecuta el notebook Jupyter para ver el análisis completo:

```bash
jupyter notebook notebooks/EDA_Análisis_Exploratorio.ipynb
```

## 📝 Requisitos del Proyecto (CRISP-DM)

### ✅ Fase 1: Comprensión del Negocio (20 puntos)
- [ ] Documentación del problema de negocio

### ✅ Fase 2: Comprensión de los Datos (10 puntos)
- [x] Dataset sintético con 30,000 registros
- [x] Análisis exploratorio completo (EDA)

### ✅ Fase 3: Preparación de Datos (20 puntos)
- [x] Pipeline de limpieza
- [x] Feature engineering
- [x] Transformaciones
- [x] División temporal train/val/test

### ✅ Fase 4: Modelado con MLflow (20 puntos)
- [x] Configuración de MLflow
- [x] 4 algoritmos diferentes
- [x] Hyperparameter tuning
- [x] Logging de parámetros y métricas
- [x] Model Registry

### ✅ Fase 5: Evaluación (10 puntos)
- [x] Métricas completas
- [x] Matriz de confusión
- [x] Curvas ROC y Precision-Recall
- [x] Feature importance

### ⏳ Fase 6: Presentación (20 puntos)
- [ ] Presentación ejecutiva (máximo 20 slides)

## 🛠️ Tecnologías Utilizadas

- **Python 3.11+**
- **Pandas** - Manipulación de datos
- **Scikit-learn** - Machine Learning
- **XGBoost** - Gradient Boosting
- **LightGBM** - Gradient Boosting rápido
- **MLflow** - Experiment tracking y Model Registry
- **Matplotlib/Seaborn** - Visualización

## 📞 Contacto

Para preguntas o problemas, consulta la documentación del proyecto o contacta al equipo.

## 📄 Licencia

Este proyecto es parte del curso de Machine Learning Operations.
