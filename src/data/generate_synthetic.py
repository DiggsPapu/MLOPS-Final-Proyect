import pandas as pd
import numpy as np
import os
np.random.seed(42)

n = 10000

# Generate synthetic features
customer_id = np.arange(1, n + 1)

# Numerical features
saldo_mora = np.random.exponential(scale=5000, size=n)  # positive skew
dias_mora = np.random.randint(0, 365, size=n)
intentos_previos = np.random.poisson(lam=2, size=n)
wait_time_seconds = np.random.exponential(scale=120, size=n)  # typical wait times
call_center_load = np.clip(np.random.normal(loc=0.7, scale=0.15, size=n), 0, 1)

# Categorical features
product_type = np.random.choice(['Credito_Personal', 'Tarjeta', 'Hipoteca', 'Automotriz'], size=n, p=[0.4, 0.3, 0.2, 0.1])
segmento_riesgo = np.random.choice(['Alto', 'Medio', 'Bajo'], size=n, p=[0.3, 0.5, 0.2])
iv_menu = np.random.choice(['Pago', 'Consulta', 'Acuerdo', 'Otros'], size=n)

# Dates: call timestamp and derived features
start_date = pd.to_datetime("2024-01-01")
call_time = start_date + pd.to_timedelta(np.random.randint(0, 365, size=n), unit='D')
hora_llamada = np.random.randint(0, 24, size=n)

# Derived correlation: high wait time + high load + high risk → higher chance of abandonment
base_prob = (
    0.2 * (wait_time_seconds / wait_time_seconds.max()) +
    0.3 * (dias_mora / dias_mora.max()) +
    0.3 * (call_center_load) +
    0.2 * (segmento_riesgo == 'Alto').astype(int)
)

base_prob = (base_prob - base_prob.min()) / (base_prob.max() - base_prob.min())

# Balanced target
threshold = np.median(base_prob)
abandono = (base_prob > threshold).astype(int)

# Build DataFrame
df = pd.DataFrame({
    "customer_id": customer_id,
    "saldo_mora": saldo_mora.round(2),
    "dias_mora": dias_mora,
    "intentos_previos": intentos_previos,
    "wait_time_seconds": wait_time_seconds.round(1),
    "call_center_load": call_center_load.round(2),
    "product_type": product_type,
    "segmento_riesgo": segmento_riesgo,
    "iv_menu": iv_menu,
    "call_time": call_time,
    "hora_llamada": hora_llamada,
    "abandono": abandono
})

# Save CSV
path = os.path.join(os.getcwd(), "data","synthetic","synthetic_calls.csv")
df.to_csv(path, index=False)

df.head(), path
