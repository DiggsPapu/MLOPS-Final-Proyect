#!/bin/bash
# Script para iniciar MLflow server

echo "🚀 Iniciando MLflow Server..."
echo "📊 UI disponible en: http://localhost:5000"
echo ""

# Crear directorio para MLflow si no existe
mkdir -p mlruns

# Iniciar MLflow server
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

