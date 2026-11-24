"""Genera datos de prueba para los gráficos p, np, c y u y verifica el Módulo 3.

Cada archivo tiene 38 muestras: 30 de Fase I (con 2 causas especiales sembradas)
y 8 de Fase II (las 3 primeras en control y las 5 últimas con el proceso desplazado).
"""
import os

import numpy as np
import pandas as pd

rng = np.random.default_rng(2024)
os.makedirs("datos_prueba", exist_ok=True)
N_F1, N_F2 = 30, 8
TOTAL = N_F1 + N_F2
lotes = np.arange(1, TOTAL + 1)
CAUSAS = (7, 21)                    # posiciones (base 0) = lotes 8 y 22
FASE2_DESPLAZADA = np.arange(TOTAL - 5, TOTAL)  # últimas 5 muestras


def guardar(nombre, df):
    df.to_csv(f"datos_prueba/{nombre}", index=False)
    print(f"  creado: datos_prueba/{nombre}")


print("Generando archivos...")

# ---- p: fracción defectuosa, n variable ----
n = rng.integers(100, 151, TOTAL)
p = np.full(TOTAL, 0.06)
p[list(CAUSAS)] = 0.20
p[FASE2_DESPLAZADA] = 0.18
guardar("atributos_p.csv", pd.DataFrame(
    {"Lote": lotes, "n": n, "Defectuosos": rng.binomial(n, p)}))

# ---- np: número de defectuosos, n constante ----
n = np.full(TOTAL, 100)
p = np.full(TOTAL, 0.06)
p[list(CAUSAS)] = 0.20
p[FASE2_DESPLAZADA] = 0.20
guardar("atributos_np.csv", pd.DataFrame(
    {"Lote": lotes, "n": n, "Defectuosos": rng.binomial(n, p)}))

# ---- c: defectos, área constante ----
lam = np.full(TOTAL, 8.0)
lam[list(CAUSAS)] = 26.0
lam[FASE2_DESPLAZADA] = 22.0
guardar("atributos_c.csv", pd.DataFrame(
    {"Lote": lotes, "Defectos": rng.poisson(lam)}))

# ---- u: defectos por unidad, número de unidades variable ----
n = rng.integers(8, 15, TOTAL)
tasa = np.full(TOTAL, 1.0)
tasa[list(CAUSAS)] = 2.8
tasa[FASE2_DESPLAZADA] = 2.5
guardar("atributos_u.csv", pd.DataFrame(
    {"Lote": lotes, "Unidades": n, "Defectos": rng.poisson(n * tasa)}))

# ---- Pareto: tipos de defecto ----
guardar("pareto_defectos.csv", pd.DataFrame({
    "Tipo de defecto": ["Rayones", "Dimensión fuera de tolerancia", "Porosidad",
                        "Mal acabado", "Empaque dañado", "Otros"],
    "Frecuencia": [58, 34, 21, 12, 8, 5]}))

# ---------------------------------------------------------------- verificación
print("\nVerificando con las funciones del Módulo 3...")
try:
    from modules.m3_atributos import estimar, fase1, limites, valores
except Exception as e:  # noqa: BLE001
    print(f"No se pudo importar el módulo: {e}")
    raise SystemExit

PRUEBAS = [("p", "atributos_p.csv", "Defectuosos", "n"),
           ("np", "atributos_np.csv", "Defectuosos", "n"),
           ("c", "atributos_c.csv", "Defectos", None),
           ("u", "atributos_u.csv", "Defectos", "Unidades")]

for tipo, archivo, col_x, col_n in PRUEBAS:
    d = pd.read_csv(f"datos_prueba/{archivo}")
    x = d[col_x].to_numpy(float)
    nn = d[col_n].to_numpy(float) if col_n else np.ones(len(d))
    etq = d["Lote"].astype(str).to_numpy()
    excl, _ = fase1(tipo, x[:N_F1], nn[:N_F1], etq[:N_F1])
    m = ~np.isin(etq[:N_F1], excl)
    par = estimar(tipo, x[:N_F1][m], nn[:N_F1][m])
    y = valores(tipo, x, nn)
    lci, lc, lcs = limites(tipo, par, nn)
    fuera = np.flatnonzero((y < lci) | (y > lcs))
    f2 = [str(etq[i]) for i in fuera if i >= N_F1]
    print(f"\nGráfico {tipo}  ({archivo})")
    print(f"  Fase I: lotes excluidos automáticamente = {excl or 'ninguno'} (se esperan 8 y 22)")
    print(f"  Parámetro central tras depurar = {par:.4f}")
    print(f"  Fase II: lotes fuera de límites = {f2 or 'ninguno'} (se esperan varios entre 34 y 38)")

print("\nListo. Ahora pruebe cada archivo en la app (Módulo 3).")