"""Módulo 3 (complemento): diagrama de Pareto y diagrama de Ishikawa."""
import textwrap

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

from modules.m1_datos import cargar_archivo
from utils.graficos import AZUL, ROJO, VERDE, NARANJA, GRIS, estilo

EJEMPLO_PARETO = pd.DataFrame({
    "Tipo de defecto": ["Rayones", "Dimensión fuera de tolerancia", "Porosidad", "Mal acabado", "Otros"],
    "Frecuencia": [42, 27, 15, 9, 7]})

CATEGORIAS = ["Mano de obra", "Máquina", "Método", "Material", "Medición", "Medio ambiente"]
EJEMPLO_ISHIKAWA = pd.DataFrame({
    "Categoría": ["Mano de obra", "Mano de obra", "Máquina", "Máquina", "Método", "Material",
                  "Material", "Medición", "Medio ambiente"],
    "Causa": ["Falta de capacitación", "Cambio de turno", "Desgaste de herramienta",
              "Falta de mantenimiento", "Procedimiento no estandarizado", "Lote de materia prima distinto",
              "Proveedor sin certificar", "Instrumento sin calibrar", "Temperatura variable"]})


# ---------------------------------------------------------------- Pareto
def graf_pareto(d, umbral):
    """Barras ordenadas + línea de porcentaje acumulado. Devuelve (figura, tabla, n de vitales)."""
    d = d.sort_values("Frecuencia", ascending=False).reset_index(drop=True)
    total = float(d["Frecuencia"].sum())
    d["% del total"] = d["Frecuencia"] / total * 100
    d["% acumulado"] = d["Frecuencia"].cumsum() / total * 100
    cruce = int(np.argmax(d["% acumulado"].to_numpy() >= umbral))
    vitales = [i <= cruce for i in range(len(d))]
    cats = d["Tipo de defecto"].tolist()

    fig = make_subplots(specs=[[{"secondary_y": True}]])
    fig.add_trace(go.Bar(x=cats, y=d["Frecuencia"], name="Frecuencia",
                         marker_color=[AZUL if v else GRIS for v in vitales],
                         text=d["Frecuencia"], textposition="outside", cliponaxis=False),
                  secondary_y=False)
    fig.add_trace(go.Scatter(x=cats, y=d["% acumulado"], mode="lines+markers", name="% acumulado",
                             line=dict(color=ROJO, width=2.5), marker=dict(size=8)),
                  secondary_y=True)
    fig.add_trace(go.Scatter(x=cats, y=[umbral] * len(cats), mode="lines", name=f"Umbral {umbral} %",
                             line=dict(color=NARANJA, dash="dash")), secondary_y=True)
    fig.update_yaxes(title_text="Frecuencia", range=[0, total * 1.05], secondary_y=False)
    fig.update_yaxes(title_text="% acumulado", range=[0, 105], ticksuffix=" %", showgrid=False,
                     secondary_y=True)
    fig.update_xaxes(title_text="Tipo de defecto")
    return estilo(fig, "Diagrama de Pareto", 460), d, cruce + 1


def mostrar_pareto():
    st.caption("El Pareto ordena los tipos de defecto de mayor a menor frecuencia para identificar los "
               "«pocos vitales» que explican la mayor parte del problema. Requiere registrar el TIPO de "
               "defecto durante la inspección, no solo cuántos hubo.")
    with st.container(border=True):
        c = st.columns([1.2, 1.2, 3])
        umbral = int(c[0].number_input("Umbral acumulado (%)", 50, 95, 80, 5, key="par_u"))
        usar_archivo = c[1].checkbox("Cargar desde archivo", key="par_f")

    if usar_archivo:
        archivo = st.file_uploader("Archivo con el tipo de defecto y su frecuencia (CSV o Excel)",
                                   type=["csv", "xlsx", "xls"], key="par_up")
        if archivo is None:
            st.info("Suba un archivo o desmarque «Cargar desde archivo» para escribir los datos a mano.")
            return
        df0 = cargar_archivo(archivo)
        c = st.columns(2)
        col_cat = c[0].selectbox("Columna del tipo de defecto", df0.columns, key="par_cc")
        col_fr = c[1].selectbox("Columna de frecuencia", df0.columns, key="par_cf")
        datos = pd.DataFrame({"Tipo de defecto": df0[col_cat].astype(str),
                              "Frecuencia": pd.to_numeric(df0[col_fr], errors="coerce")})
    else:
        st.caption("Los datos de abajo son un EJEMPLO. Reemplácelos por los defectos reales de la empresa "
                   "(puede agregar filas al final de la tabla).")
        datos = st.data_editor(EJEMPLO_PARETO.copy(), num_rows="dynamic", hide_index=True, key="par_tab")

    d = datos.copy()
    d["Frecuencia"] = pd.to_numeric(d["Frecuencia"], errors="coerce")
    d = d.dropna()
    d = d[d["Frecuencia"] > 0]
    d["Tipo de defecto"] = d["Tipo de defecto"].astype(str)
    d = d.groupby("Tipo de defecto", as_index=False)["Frecuencia"].sum()
    if len(d) < 2:
        st.info("Se necesitan al menos 2 tipos de defecto con frecuencia positiva.")
        return

    fig, tabla, n_vit = graf_pareto(d, umbral)
    st.plotly_chart(fig, width="stretch", key="g_pareto")
    vit = tabla["Tipo de defecto"].iloc[:n_vit].tolist()
    acum = float(tabla["% acumulado"].iloc[n_vit - 1])
    st.success(f"Pocos vitales: **{', '.join(vit)}** ({n_vit} de {len(tabla)} tipos) concentran "
               f"**{acum:.1f} %** de los defectos. Estos son los primeros candidatos para el análisis de "
               "causas (Ishikawa) y las propuestas de mejora.")
    st.caption("Azul: pocos vitales (hasta alcanzar el umbral) | Gris: resto | El umbral de 80 % es una "
               "convención, no una ley.")
    st.dataframe(tabla.round(2), hide_index=True)
    st.session_state["pareto"] = tabla
    st.session_state["pareto_top"] = str(tabla["Tipo de defecto"].iloc[0])


# ---------------------------------------------------------------- Ishikawa
def _envolver(texto, ancho):
    return "<br>".join(textwrap.wrap(str(texto), ancho)) or " "


def graf_ishikawa(efecto, causas):
    """Diagrama 6M con composición original y causas en cajas independientes."""
    fig = go.Figure()
    sup = ["Máquina", "Método", "Material"]
    inf = ["Mano de obra", "Medición", "Medio ambiente"]
    atts, dx, ytip = [40, 68, 96], 16, 42
    fig.add_shape(type="line", x0=0, y0=0, x1=100, y1=0,
                  line=dict(color=AZUL, width=5))
    fig.add_shape(type="rect", x0=100, x1=122, y0=-10, y1=10,
                  line=dict(color=AZUL, width=2), fillcolor="rgba(40,127,196,.12)")
    fig.add_annotation(x=111, y=0, text=f"<b>{_envolver(efecto, 16)}</b>",
                       showarrow=False, font=dict(size=12, color="#263746"))
    for fila, nombres in ((1, sup), (-1, inf)):
        for ax, cat in zip(atts, nombres):
            tx, ty = ax - dx, fila * ytip
            fig.add_shape(type="line", x0=tx, y0=ty, x1=ax, y1=0,
                          line=dict(color="#7A8B99", width=2.5))
            fig.add_annotation(x=tx, y=ty + fila * 2, text=f"<b>{cat}</b>",
                               showarrow=False, yanchor="bottom" if fila == 1 else "top",
                               bgcolor="#287FC4", bordercolor="#1D639C",
                               font=dict(color="white", size=12), borderpad=5)
            lista = [str(c).strip() for c in causas.get(cat, []) if str(c).strip()]
            n = min(len(lista), 5)
            for i, causa in enumerate(lista[:5]):
                t = (i + 1) / (n + 1)
                px, py = tx + t * (ax - tx), ty * (1 - t)
                bx, by = px - 13, py + (4 if fila == 1 else -4)
                fig.add_shape(type="rect", x0=bx - 12, x1=bx + 12,
                              y0=by - 3.5, y1=by + 3.5,
                              line=dict(color="#B8C7D3", width=1), fillcolor="#F2F7FB")
                fig.add_annotation(x=bx, y=by, text=_envolver(causa, 20),
                                   showarrow=False, font=dict(size=9, color="#34495A"),
                                   align="center")
    fig.update_xaxes(visible=False, range=[-3, 124])
    fig.update_yaxes(visible=False, range=[-58, 58])
    fig.update_layout(height=600, margin=dict(l=8, r=8, t=18, b=8),
                      showlegend=False, paper_bgcolor="white", plot_bgcolor="white")
    return fig


def mostrar_ishikawa():
    st.caption("El Ishikawa (causa-efecto o espina de pescado) organiza las posibles causas de un problema "
               "en las 6M. Se construye con una lluvia de ideas con el personal de la empresa; luego se "
               "verifican con datos las causas más probables.")
    efecto = st.text_input("Efecto o problema (cabeza del pez)",
                           value=st.session_state.get("pareto_top", "Alta fracción de no conformes"))
    st.caption("Si ya construyó el Pareto, aquí se propone su defecto principal. Puede cambiarlo.")
    st.caption("Los datos de abajo son un EJEMPLO: reemplácelos por las causas identificadas en la visita. "
               "Para que el gráfico se lea bien, use máximo 6 causas por rama.")
    tabla = st.data_editor(
        EJEMPLO_ISHIKAWA.copy(), num_rows="dynamic", hide_index=True, key="ish_tab",
        column_config={"Categoría": st.column_config.SelectboxColumn(options=CATEGORIAS, required=True),
                       "Causa": st.column_config.TextColumn(width="large")})
    causas = {c: [] for c in CATEGORIAS}
    for _, f in tabla.dropna().iterrows():
        if f["Categoría"] in causas and str(f["Causa"]).strip():
            causas[f["Categoría"]].append(str(f["Causa"]).strip())
    st.plotly_chart(graf_ishikawa(efecto, causas), width="stretch", key="g_ishikawa")
    st.caption("Para guardar el diagrama como imagen use el ícono de cámara arriba a la derecha del gráfico.")
    st.session_state["ishikawa"] = {"efecto": efecto, "causas": causas}