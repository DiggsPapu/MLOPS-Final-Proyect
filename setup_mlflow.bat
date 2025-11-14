@echo off
REM Script para iniciar MLflow server en Windows

echo 🚀 Iniciando MLflow Server...
echo 📊 UI disponible en: http://localhost:5000
echo.

REM Crear directorio para MLflow si no existe
if not exist mlruns mkdir mlruns

REM Iniciar MLflow server
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port 5000

pause

