# ===============================================================
# 📌 SCRIPT DE VERIFICACIÓN DE CONFIGURACIÓN
# ===============================================================

"""Verificar que todo esté configurado correctamente"""

import sys
from pathlib import Path

def check_imports():
    """Verificar que todas las dependencias estén instaladas"""
    print("🔍 Verificando dependencias...")
    
    required_packages = [
        'pandas', 'numpy', 'sklearn', 
        'xgboost', 'lightgbm', 'mlflow',
        'matplotlib', 'seaborn'
    ]
    
    missing = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {package}")
        except ImportError:
            print(f"  ❌ {package} - NO INSTALADO")
            missing.append(package)
    
    if missing:
        print(f"\n⚠️ Faltan las siguientes dependencias: {', '.join(missing)}")
        print("   Ejecuta: pip install -r requirements.txt")
        return False
    
    print("\n✅ Todas las dependencias están instaladas")
    return True


def check_data():
    """Verificar que los datos existan"""
    print("\n🔍 Verificando datos...")
    
    project_root = Path(__file__).parent
    data_path = project_root / "data" / "synthetic" / "synthetic_calls.csv"
    
    if data_path.exists():
        print(f"  ✅ Datos encontrados: {data_path}")
        return True
    else:
        print(f"  ❌ Datos no encontrados: {data_path}")
        print("   Ejecuta: python src/data/generate_synthetic.py")
        return False


def check_mlflow():
    """Verificar conexión con MLflow"""
    print("\n🔍 Verificando MLflow...")
    
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        
        # Intentar listar experimentos
        try:
            experiments = mlflow.search_experiments()
            print(f"  ✅ MLflow server conectado")
            print(f"  📊 Experimentos encontrados: {len(experiments)}")
            return True
        except Exception as e:
            print(f"  ⚠️ MLflow server no está corriendo")
            print(f"     Error: {e}")
            print("     Ejecuta: setup_mlflow.bat (Windows) o ./setup_mlflow.sh (Linux/Mac)")
            return False
    except ImportError:
        print("  ❌ MLflow no está instalado")
        return False


def check_structure():
    """Verificar estructura de directorios"""
    print("\n🔍 Verificando estructura de directorios...")
    
    project_root = Path(__file__).parent
    required_dirs = [
        "src/data",
        "src/models",
        "data/synthetic",
        "notebooks"
    ]
    
    all_exist = True
    for dir_path in required_dirs:
        full_path = project_root / dir_path
        if full_path.exists():
            print(f"  ✅ {dir_path}")
        else:
            print(f"  ❌ {dir_path} - NO EXISTE")
            all_exist = False
    
    return all_exist


def main():
    """Ejecutar todas las verificaciones"""
    print("="*70)
    print("🔍 VERIFICACIÓN DE CONFIGURACIÓN DEL PROYECTO")
    print("="*70)
    
    checks = [
        ("Estructura", check_structure),
        ("Dependencias", check_imports),
        ("Datos", check_data),
        ("MLflow", check_mlflow)
    ]
    
    results = []
    for name, check_func in checks:
        result = check_func()
        results.append((name, result))
    
    print("\n" + "="*70)
    print("📊 RESUMEN")
    print("="*70)
    
    all_ok = True
    for name, result in results:
        status = "✅ OK" if result else "❌ FALTA"
        print(f"  {name:20s}: {status}")
        if not result:
            all_ok = False
    
    if all_ok:
        print("\n✅ ¡Todo está configurado correctamente!")
        print("   Puedes ejecutar: python src/main.py")
    else:
        print("\n⚠️ Hay problemas de configuración. Revisa los mensajes arriba.")
        sys.exit(1)


if __name__ == "__main__":
    main()

