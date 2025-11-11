import pandas as pd
import numpy as np
import os
np.random.seed(42)

# -----------------------------
# PARAMETERS
# -----------------------------
n = 30000

# -----------------------------
# BASIC CUSTOMER FEATURES
# -----------------------------
customer_id = np.arange(1, n + 1)

edad = np.random.normal(loc=42, scale=12, size=n).astype(int)
edad = np.clip(edad, 18, 85)

ingreso_mensual = np.random.lognormal(mean=8, sigma=0.5, size=n).round(2)

region = np.random.choice(
    ["Norte", "Sur", "Centro", "Occidente", "Oriente"],
    size=n,
    p=[0.2, 0.15, 0.4, 0.15, 0.1]
)

# -----------------------------
# NUMERIC FINANCIAL FEATURES
# -----------------------------
saldo_mora = np.random.exponential(scale=6000, size=n)
dias_mora = np.random.randint(0, 365, size=n)
intentos_previos = np.random.poisson(lam=2, size=n)
call_center_load = np.clip(np.random.normal(loc=0.7, scale=0.2, size=n), 0, 1)

# -----------------------------
# CATEGORICAL FEATURES
# -----------------------------
product_type = np.random.choice(
    ["Credito_Personal", "Tarjeta", "Hipoteca", "Automotriz", "Pyme"],
    size=n,
    p=[0.35, 0.3, 0.15, 0.1, 0.1]
)

segmento_riesgo = np.random.choice(
    ["Alto", "Medio", "Bajo"],
    size=n,
    p=[0.30, 0.50, 0.20]
)

iv_menu = np.random.choice(
    ["Pago", "Consulta", "Acuerdo", "Otros"],
    size=n
)

score_crediticio = np.random.normal(loc=600, scale=100, size=n).clip(300, 850).astype(float)

# Tecnologías de canal
canal_contacto = np.random.choice(
    ["WhatsApp", "Llamada", "Email", "App"],
    size=n,
    p=[0.25, 0.5, 0.15, 0.1]
)

# -----------------------------
# DATE FEATURES
# -----------------------------
start_date = pd.to_datetime("2024-01-01")
call_time = start_date + pd.to_timedelta(np.random.randint(0, 365, size=n), unit="D")
hora_llamada = np.random.randint(0, 24, size=n)

# Fecha en distinto formato (string)
fecha_creacion = (
    start_date + pd.to_timedelta(np.random.randint(0, 730, size=n), unit="D")
).strftime("%Y-%m-%d")  # base default

# Convert randomly to mixed formats
formatos = ["%Y/%m/%d", "%d-%m-%Y", "%m/%d/%Y"]
mask_fmt = np.random.choice(len(formatos), size=n)
fecha_creacion = [
    pd.to_datetime(fecha_creacion[i]).strftime(formatos[mask_fmt[i]])
    for i in range(n)
]

# -----------------------------
# ADDITIONAL FEATURES
# -----------------------------
wait_time_seconds = np.random.exponential(scale=120, size=n)
wait_time_seconds[np.random.choice(n, 200)] *= 5  # outliers

educacion = np.random.choice(
    ["Primaria", "Secundaria", "Universitaria", "Postgrado"],
    size=n,
    p=[0.2, 0.4, 0.3, 0.1]
)

historial_pagos = np.random.choice(["Excelente", "Bueno", "Regular", "Malo"], size=n)

timezone = np.random.choice(["GMT-6", "GMT-5", "GMT-7"], size=n)

# Feature adicional: monto del último pago
ultimo_pago = (np.random.exponential(scale=2000, size=n)).round(2)

# Feature adicional: días desde último pago
dias_ultimo_pago = np.random.randint(0, 400, size=n)

# Feature dummy: campaña asignada
campaña = np.random.choice(["A", "B", "C", "D"], size=n)

# -----------------------------
# CORRELATED FEATURES
# -----------------------------
base_prob = (
    0.15 * (saldo_mora / saldo_mora.max()) +
    0.15 * (dias_mora / dias_mora.max()) +
    0.2 * (call_center_load) +
    0.1 * (segmento_riesgo == "Alto").astype(float) +
    0.1 * (product_type == "Credito_Personal").astype(float) +
    0.1 * (score_crediticio < 550).astype(float) +
    0.1 * (canal_contacto == "Llamada").astype(float) +
    0.05 * (intentos_previos / np.max(intentos_previos)) +
    0.05 * (dias_ultimo_pago / dias_ultimo_pago.max())
)

# Normalize
base_prob = (base_prob - base_prob.min()) / (base_prob.max() - base_prob.min())

# Initial target
thr = np.median(base_prob)
abandono = (base_prob > thr).astype(int)

# -----------------------------
# ADD CONTROLLED NOISE TO TARGET
# -----------------------------
noise_rate = 0.10
noise_idx = np.random.choice(n, size=int(n * noise_rate), replace=False)

abandono_noisy = abandono.copy()
abandono_noisy[noise_idx] = 1 - abandono_noisy[noise_idx]  # flip 0<->1
abandono = abandono_noisy

# -----------------------------
# INTENTIONAL ERRORS + MISSING
# -----------------------------
# Some invalid channel
canal_contacto[np.random.choice(n, 30)] = "???"

# Some invalid mora days
dias_mora[np.random.choice(n, 50)] = -1

# Missing values
for col in ["ingreso_mensual", "score_crediticio", "historial_pagos"]:
    idx = np.random.choice(n, size=int(n * 0.05), replace=False)
    if col == "ingreso_mensual":
        ingreso_mensual[idx] = np.nan
    elif col == "score_crediticio":
        score_crediticio[idx] = np.nan
    elif col == "historial_pagos":
        historial_pagos[idx] = None

# -----------------------------
# BUILD DATAFRAME
# -----------------------------
df = pd.DataFrame({
    "customer_id": customer_id,
    "edad": edad,
    "ingreso_mensual": ingreso_mensual,
    "region": region,
    "saldo_mora": saldo_mora.round(2),
    "dias_mora": dias_mora,
    "dias_ultimo_pago": dias_ultimo_pago,
    "ultimo_pago": ultimo_pago,
    "intentos_previos": intentos_previos,
    "wait_time_seconds": wait_time_seconds.round(1),
    "call_center_load": call_center_load.round(2),
    "product_type": product_type,
    "segmento_riesgo": segmento_riesgo,
    "iv_menu": iv_menu,
    "score_crediticio": score_crediticio,
    "canal_contacto": canal_contacto,
    "call_time": call_time,
    "hora_llamada": hora_llamada,
    "fecha_creacion": fecha_creacion,
    "educacion": educacion,
    "historial_pagos": historial_pagos,
    "timezone": timezone,
    "campaña": campaña,
    "abandono": abandono
})

# -----------------------------
# SAVE
# -----------------------------
path = os.path.join(os.getcwd(), "data", "synthetic", "synthetic_calls.csv")
os.makedirs(os.path.dirname(path), exist_ok=True)
df.to_csv(path, index=False)

df.head(), path
