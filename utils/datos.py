"""Funciones compartidas por varios módulos para preparar los datos."""
import numpy as np
import pandas as pd
import streamlit as st


def tipo_grafico_por_defecto(df, col_valor):
    """Elige el gráfico de variables según el tamaño de los subgrupos."""
    n = int(df.groupby("_subgrupo")[col_valor].count().max())
    if n == 1:
        return "I-MR"
    return "X̄-R" if n <= 10 else "X̄-S"


def datos_estables(df, usar_exclusion=True):
    """Devuelve los datos sin los puntos retirados en la Fase I del Módulo 2."""
    lim = st.session_state.get("limites")
    if not usar_exclusion or not lim or not lim.get("excluidos"):
        return df
    excl = lim["excluidos"]
    if lim["tipo"] == "I-MR":
        mascara = ~pd.Series(np.arange(1, len(df) + 1)).isin(excl).values
    else:
        mascara = ~df["_subgrupo"].isin(excl).values
    return df[mascara]