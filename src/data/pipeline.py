# ===============================================================
# 📌 PIPELINE DE PREPARACIÓN DE DATOS
# ===============================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.base import BaseEstimator, TransformerMixin
import datetime as dt
import os


# ===============================================================
# ✅ 1) Cargar datos
# ===============================================================
path = os.path.join(os.getcwd(), "data", "synthetic", "synthetic_calls.csv")
df = pd.read_csv(path)

print("\n✅ Datos cargados")
print(df.head())


# ===============================================================
# ✅ 2) Limpieza básica
# ===============================================================

# Eliminar duplicados
df.drop_duplicates(inplace=True)

# Convertir columnas de fecha a datetime
date_cols = ["call_time","fecha_alta_cliente","fecha_nacimiento"]
for c in date_cols:
    if c in df.columns:
        df[c] = pd.to_datetime(df[c], errors="coerce")   # Coerce → genera NaT si error

# -------------------------
# Revisión de nulos
# -------------------------
print("\n📌 % Nulos por columna")
print(df.isna().mean().sort_values(ascending=False))


# ===============================================================
# ✅ 3) Feature Engineering
# ===============================================================

# Edad
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

# Día, mes y día de la semana
if "call_time" in df.columns:
    df["call_year"]  = df["call_time"].dt.year
    df["call_month"] = df["call_time"].dt.month
    df["call_day"]   = df["call_time"].dt.day
    df["call_wday"]  = df["call_time"].dt.weekday


# ===============================================================
# ✅ 4) Manejo de outliers (winsorizing suave)
# ===============================================================
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
num_cols.remove("abandono")   # No tocar target

for col in num_cols:
    q1, q99 = df[col].quantile([0.01, 0.99])
    df[col] = df[col].clip(q1, q99)


# ===============================================================
# ✅ 5) Validación de calidad
# ===============================================================
print("\n🔎 Validando calidad…")

# % nulos permitido
for col in df.columns:
    if df[col].isna().mean() > 0.3:
        print(f"⚠️ ALERTA: {col} tiene +30% valores nulos")

if df["abandono"].value_counts(normalize=True).max() > 0.6:
    print("⚠️ ALERTA: Target desbalanceado")
else:
    print("✅ Target balanceado")


# ===============================================================
# ✅ 6) División temporal train/val/test
# ===============================================================

df = df.sort_values("call_time")

train = df[df["call_time"] < "2024-07-01"]
val   = df[(df["call_time"] >= "2024-07-01") & (df["call_time"] < "2024-10-01")]
test  = df[df["call_time"] >= "2024-10-01"]

print("\n📌 División temporal:")
print(f"Train: {train.shape}")
print(f"Val:   {val.shape}")
print(f"Test:  {test.shape}")

X_train = train.drop(columns=["abandono"])
y_train = train["abandono"]

X_val = val.drop(columns=["abandono"])
y_val = val["abandono"]

X_test = test.drop(columns=["abandono"])
y_test = test["abandono"]


# ===============================================================
# ✅ 7) Column types
# ===============================================================

num_features = X_train.select_dtypes(include=[np.number]).columns
cat_features = X_train.select_dtypes(exclude=[np.number, "datetime64"]).columns

print("\n🔢 Numéricas:", list(num_features))
print("🔤 Categóricas:", list(cat_features))


# ===============================================================
# ✅ 8) Transformaciones (Scaler + One-Hot)
# ===============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler())
])

categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, num_features),
        ("cat", categorical_transformer, cat_features),
    ]
)


# ===============================================================
# ✅ 9) Pipeline final
# ===============================================================
pipeline = Pipeline(steps=[
    ("preprocess", preprocessor)
])


# ===============================================================
# ✅ 10) Fit / Transform
# ===============================================================
X_train_proc = pipeline.fit_transform(X_train)
X_val_proc   = pipeline.transform(X_val)
X_test_proc  = pipeline.transform(X_test)

print("\n✅ Transformación completa")
print("Train procesado →", X_train_proc.shape)
print("Val procesado   →", X_val_proc.shape)
print("Test procesado  →", X_test_proc.shape)
