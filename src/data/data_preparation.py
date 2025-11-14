# ===============================================================
# 📌 MÓDULO DE PREPARACIÓN DE DATOS
# ===============================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
import os
from pathlib import Path


def load_data(data_path=None):
    """Cargar datos desde CSV"""
    if data_path is None:
        project_root = Path(__file__).parent.parent.parent
        data_path = project_root / "data" / "synthetic" / "synthetic_calls.csv"
    
    df = pd.read_csv(data_path)
    print(f"✅ Datos cargados: {df.shape[0]:,} filas × {df.shape[1]} columnas")
    return df


def clean_data(df):
    """Limpieza básica de datos"""
    df = df.copy()
    
    # Eliminar duplicados
    df.drop_duplicates(inplace=True)
    
    # Convertir columnas de fecha a datetime
    date_cols = ["call_time", "fecha_alta_cliente", "fecha_nacimiento"]
    for c in date_cols:
        if c in df.columns:
            df[c] = pd.to_datetime(df[c], errors="coerce")
    
    # Limpiar valores inválidos
    if "canal_contacto" in df.columns:
        df["canal_contacto"] = df["canal_contacto"].replace("???", np.nan)
    
    if "dias_mora" in df.columns:
        df["dias_mora"] = df["dias_mora"].replace(-1, np.nan)
    
    return df


def feature_engineering(df):
    """Feature engineering"""
    df = df.copy()
    
    # Edad desde fecha de nacimiento
    if "fecha_nacimiento" in df.columns:
        df["edad"] = (
            (pd.to_datetime("today") - df["fecha_nacimiento"])
            / pd.Timedelta(days=365.25)
        ).round()
    
    # Antigüedad del cliente
    if "fecha_alta_cliente" in df.columns:
        df["antiguedad_cliente"] = (
            (pd.to_datetime("today") - df["fecha_alta_cliente"])
            / pd.Timedelta(days=365.25)
        ).round()
    
    # Features temporales
    if "call_time" in df.columns:
        df["call_year"] = df["call_time"].dt.year
        df["call_month"] = df["call_time"].dt.month
        df["call_day"] = df["call_time"].dt.day
        df["call_wday"] = df["call_time"].dt.weekday
    
    return df


def handle_outliers(df):
    """Manejo de outliers usando winsorizing"""
    df = df.copy()
    num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if "abandono" in num_cols:
        num_cols.remove("abandono")
    
    for col in num_cols:
        q1, q99 = df[col].quantile([0.01, 0.99])
        df[col] = df[col].clip(q1, q99)
    
    return df


def split_temporal(df, train_date="2024-07-01", val_date="2024-10-01"):
    """División temporal de datos"""
    df = df.sort_values("call_time")
    
    train = df[df["call_time"] < train_date]
    val = df[(df["call_time"] >= train_date) & (df["call_time"] < val_date)]
    test = df[df["call_time"] >= val_date]
    
    print(f"\n📌 División temporal:")
    print(f"  Train: {train.shape[0]:,} registros")
    print(f"  Val:   {val.shape[0]:,} registros")
    print(f"  Test:  {test.shape[0]:,} registros")
    
    return train, val, test


def create_preprocessor(X_train):
    """Crear pipeline de preprocesamiento"""
    num_features = X_train.select_dtypes(include=[np.number]).columns
    cat_features = X_train.select_dtypes(exclude=[np.number, "datetime64"]).columns
    
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    
    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, num_features),
            ("cat", categorical_transformer, cat_features),
        ]
    )
    
    return preprocessor


def prepare_data(data_path=None):
    """Pipeline completo de preparación de datos"""
    # Cargar
    df = load_data(data_path)
    
    # Limpiar
    df = clean_data(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Outliers
    df = handle_outliers(df)
    
    # División temporal
    train, val, test = split_temporal(df)
    
    # Separar features y target
    X_train = train.drop(columns=["abandono"])
    y_train = train["abandono"]
    
    X_val = val.drop(columns=["abandono"])
    y_val = val["abandono"]
    
    X_test = test.drop(columns=["abandono"])
    y_test = test["abandono"]
    
    # Crear preprocessor
    preprocessor = create_preprocessor(X_train)
    
    # Transformar
    X_train_proc = preprocessor.fit_transform(X_train)
    X_val_proc = preprocessor.transform(X_val)
    X_test_proc = preprocessor.transform(X_test)
    
    print(f"\n✅ Preparación completa")
    print(f"  Train procesado: {X_train_proc.shape}")
    print(f"  Val procesado:   {X_val_proc.shape}")
    print(f"  Test procesado:  {X_test_proc.shape}")
    
    return {
        "X_train": X_train_proc,
        "y_train": y_train,
        "X_val": X_val_proc,
        "y_val": y_val,
        "X_test": X_test_proc,
        "y_test": y_test,
        "preprocessor": preprocessor,
        "feature_names": list(X_train.columns)
    }

