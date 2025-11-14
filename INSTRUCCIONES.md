# 📋 Instrucciones de Uso - Proyecto MLOps

## 🚀 Inicio Rápido

### Paso 1: Instalar Dependencias

```bash
pip install -r requirements.txt
```

### Paso 2: Generar Datos (si no existen)

```bash
python src/data/generate_synthetic.py
```

### Paso 3: Iniciar MLflow Server

**En Windows:**
```bash
setup_mlflow.bat
```

**En Linux/Mac:**
```bash
chmod +x setup_mlflow.sh
./setup_mlflow.sh
```

**O manualmente:**
```bash
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000
```

### Paso 4: Ejecutar Pipeline Completo

En una nueva terminal (con MLflow corriendo):

```bash
python src/main.py
```

## 📊 Acceder a MLflow UI

Una vez que MLflow esté corriendo, abre tu navegador en:

**http://localhost:5000**

Aquí podrás ver:
- Todos los experimentos
- Comparación de modelos
- Métricas y parámetros
- Modelos registrados en el Model Registry

## 🔍 Ejecutar Componentes Individuales

### Solo Preparación de Datos

```python
from src.data.data_preparation import prepare_data
data_dict = prepare_data()
```

### Solo Entrenamiento

```python
from src.data.data_preparation import prepare_data
from src.models.model_training import train_all_models

data_dict = prepare_data()
results = train_all_models(data_dict)
```

### Solo Evaluación

```python
from src.data.data_preparation import prepare_data
from src.models.model_evaluation import load_best_model, evaluate_on_all_splits

data_dict = prepare_data()
model, _ = load_best_model()
results = evaluate_on_all_splits(model, data_dict)
```

## 📈 Ver Resultados

Los resultados de evaluación se guardan en:
- `results/evaluation/test/` - Resultados en test set
- `results/evaluation/val/` - Resultados en validation set

Cada carpeta contiene:
- `confusion_matrix.png` - Matriz de confusión
- `roc_curve.png` - Curva ROC
- `precision_recall_curve.png` - Curva Precision-Recall
- `feature_importance.png` - Importancia de features

## 🐛 Solución de Problemas

### Error: "MLflow server no está corriendo"

Asegúrate de tener MLflow server corriendo antes de ejecutar el entrenamiento.

### Error: "No se encuentra el modelo"

Verifica que hayas ejecutado el entrenamiento primero. El modelo se registra automáticamente después del entrenamiento.

### Error: "ModuleNotFoundError"

Asegúrate de haber instalado todas las dependencias:
```bash
pip install -r requirements.txt
```

### Error: "No se puede conectar a MLflow"

Verifica que el servidor esté corriendo en `http://localhost:5000` y que no haya otro proceso usando el puerto 5000.

## 📝 Notas Importantes

1. **MLflow debe estar corriendo** antes de ejecutar el entrenamiento
2. El entrenamiento puede tardar varios minutos (especialmente con hyperparameter tuning)
3. Los resultados se guardan automáticamente en MLflow y en la carpeta `results/`
4. El mejor modelo se registra automáticamente en el Model Registry

## 🎯 Para la Presentación

1. Ejecuta el pipeline completo
2. Toma screenshots de:
   - MLflow UI mostrando los experimentos
   - Comparación de modelos
   - Model Registry con el modelo registrado
   - Métricas y gráficos de evaluación
3. Incluye estos screenshots en tu presentación

