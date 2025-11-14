# 📤 Guía para Subir el Proyecto a GitHub

## ✅ ¿Puedo subir notebooks con outputs?

**¡SÍ!** Los notebooks de Jupyter con outputs se pueden subir a GitHub y se visualizarán correctamente con todos los resultados, gráficos y métricas.

## 📋 Pasos para Subir a GitHub

### 1. Preparar el Repositorio

```bash
# Inicializar git (si no lo has hecho)
git init

# Agregar todos los archivos
git add .

# Hacer commit inicial
git commit -m "Initial commit: Proyecto MLOps con CRISP-DM"
```

### 2. Crear Repositorio en GitHub

1. Ve a [GitHub](https://github.com)
2. Crea un nuevo repositorio
3. **NO** inicialices con README, .gitignore o licencia (ya los tenemos)

### 3. Conectar y Subir

```bash
# Agregar remote
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git

# Subir código
git branch -M main
git push -u origin main
```

## 📁 Archivos que SÍ debes subir

✅ **SÍ subir:**
- `notebooks/*.ipynb` - **Con outputs incluidos** (para que se vean los resultados)
- `src/` - Todo el código fuente
- `data/` - Datos sintéticos (si no son muy grandes)
- `requirements.txt`
- `README.md`
- `INSTRUCCIONES.md`
- `.gitignore`

## ⚠️ Archivos que NO debes subir

❌ **NO subir:**
- `mlruns/` - Resultados de MLflow (muy grandes, se regeneran)
- `mlflow.db` - Base de datos de MLflow
- `venv/` o `env/` - Entorno virtual
- `__pycache__/` - Cache de Python
- `results/` - Resultados temporales (opcional, puedes subirlos si quieres)

## 🎯 Recomendaciones

### Para el README en GitHub

Agrega una sección al README con:

```markdown
## 🚀 Inicio Rápido

1. Clonar repositorio
2. Instalar dependencias: `pip install -r requirements.txt`
3. Generar datos: `python src/data/generate_synthetic.py`
4. Iniciar MLflow: `setup_mlflow.bat` (Windows) o `./setup_mlflow.sh` (Linux/Mac)
5. Ejecutar pipeline: `python src/main.py`
```

### Visualización de Notebooks

GitHub renderiza automáticamente los notebooks. Los outputs (gráficos, tablas, métricas) se verán directamente en GitHub.

### Si el repositorio es muy grande

Si los notebooks con outputs son muy grandes (>50MB), considera:

1. **Opción 1:** Usar [Git LFS](https://git-lfs.github.com/) para archivos grandes
2. **Opción 2:** Limpiar outputs de notebooks muy grandes antes de subir
3. **Opción 3:** Subir notebooks sin outputs y documentar cómo ejecutarlos

## 📊 Estructura Recomendada para GitHub

```
MLOPS-Final-Proyect/
├── .gitignore              ✅
├── README.md               ✅ (con badges, instrucciones)
├── requirements.txt        ✅
├── INSTRUCCIONES.md        ✅
├── notebooks/
│   ├── EDA_Análisis_Exploratorio.ipynb    ✅ (con outputs)
│   └── Modelado_y_Evaluacion.ipynb        ✅ (con outputs)
├── src/
│   ├── data/
│   │   ├── data_preparation.py    ✅
│   │   └── generate_synthetic.py  ✅
│   ├── models/
│   │   ├── model_training.py      ✅
│   │   └── model_evaluation.py    ✅
│   └── main.py                    ✅
├── data/
│   └── synthetic/
│       └── synthetic_calls.csv    ✅ (opcional, si no es muy grande)
└── setup_mlflow.bat / .sh         ✅
```

## 🎨 Mejoras para GitHub

### Agregar Badges al README

```markdown
![Python](https://img.shields.io/badge/python-3.11+-blue.svg)
![MLflow](https://img.shields.io/badge/MLflow-2.8+-orange.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
```

### Agregar Sección de Screenshots

```markdown
## 📸 Screenshots

### MLflow UI
![MLflow](screenshots/mlflow.png)

### Resultados del Modelo
![Results](screenshots/results.png)
```

## ✅ Checklist Antes de Subir

- [ ] Revisar `.gitignore` está configurado
- [ ] Notebooks tienen outputs (para visualización)
- [ ] README está completo y actualizado
- [ ] No hay datos sensibles en los notebooks
- [ ] `requirements.txt` está actualizado
- [ ] Código está comentado y documentado
- [ ] No hay archivos temporales o de cache

## 🚀 Comandos Útiles

```bash
# Ver qué se va a subir
git status

# Ver tamaño de archivos
du -sh *

# Limpiar outputs de notebooks (si es necesario)
pip install nbstripout
nbstripout notebooks/*.ipynb

# Verificar que .gitignore funciona
git check-ignore -v mlruns/
```

## 📝 Nota Final

**Los notebooks con outputs son perfectos para GitHub** porque:
- ✅ Permiten ver resultados sin ejecutar código
- ✅ GitHub los renderiza automáticamente
- ✅ Son ideales para documentación y presentación
- ✅ Otros pueden entender mejor el proyecto

¡Sube tus notebooks con outputs sin problema! 🎉

