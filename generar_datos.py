"""Genera datos simulados para probar el software CEP."""
import os
import numpy as np
import pandas as pd

np.random.seed(42)  # misma semilla = mismos datos cada vez

N_SUBGRUPOS = 30
TAMANO = 5
MEDIA = 50.0
SIGMA = 2.0

filas = []
for sg in range(1, N_SUBGRUPOS + 1):
    media_sg = MEDIA
    # Causas especiales a propósito: subgrupos 12 y 23 desplazados
    if sg in (12, 23):
        media_sg = MEDIA + 6
    valores = np.random.normal(media_sg, SIGMA, TAMANO)
    for v in valores:
        filas.append({"Subgrupo": sg, "Peso_g": round(v, 2)})

df = pd.DataFrame(filas)
os.makedirs("datos_prueba", exist_ok=True)
df.to_csv("datos_prueba/datos_prueba.csv", index=False)
df.to_excel("datos_prueba/datos_prueba.xlsx", index=False)
print(f"Listo: {len(df)} filas generadas en la carpeta datos_prueba")