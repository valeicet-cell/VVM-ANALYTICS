"""Pantalla de bienvenida y panel de resultados de VVM Analytics."""
import numpy as np
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.estilos import logo_uri
from utils.graficos import AZUL, ROJO, VERDE, estilo

MODULOS = [
    ("🗄️", "Datos", "Carga y organiza variables y atributos", "Datos"),
    ("📈", "Gráficos por Variables", "X̄-R, X̄-S, I-MR y más", "Gráficos por Variables"),
    ("📊", "Gráficos por Atributos", "p, np, c y u", "Gráficos por Atributos"),
    ("🎯", "Capacidad", "Cp, Cpk y desempeño del proceso", "Capacidad"),
    ("✅", "Supuestos", "Normalidad, varianza e independencia", "Supuestos"),
    ("⚙️", "Diseño de Gráficos", "ARL0, ARL1 y ATS", "Diseño de Gráficos"),
    ("📦", "Muestreo de Aceptación", "Planes y curvas OC", "Muestreo de Aceptación"),
    ("📄", "Reportes", "Informes y exportaciones", "Reportes"),
]


def ir_a(pagina):
    st.session_state["pagina"] = pagina


def kpi(titulo, valor, color, sub):
    return (f'<div class="kpi {color}"><div class="titulo">{titulo}</div>'
            f'<div class="valor">{valor}</div><div class="sub">{sub}</div></div>')


def _graf_medias(df, col):
    med = df.groupby("_subgrupo")[col].mean()
    fig = go.Figure(go.Scatter(x=list(range(1, len(med) + 1)), y=med.values,
        mode="lines+markers", name="Media del subgrupo", line=dict(color=AZUL, width=2)))
    fig.add_hline(y=float(df[col].mean()), line_color=VERDE,
                  annotation_text=f"Media global = {df[col].mean():.2f}")
    fig.update_xaxes(title_text="Subgrupo")
    fig.update_yaxes(title_text=col)
    return estilo(fig, "Resumen de medias por subgrupo", 310, derecha=100)


def _graf_hist(x, col):
    x = np.asarray(x, dtype=float)
    fig = go.Figure(go.Histogram(x=x, histnorm="probability density", name="Datos"))
    if len(x) > 1 and np.std(x, ddof=1) > 0:
        xs = np.linspace(x.min(), x.max(), 200)
        fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, x.mean(), x.std(ddof=1)),
            mode="lines", name="Normal de referencia", line=dict(color=ROJO, width=2.5)))
    fig.update_xaxes(title_text=col)
    fig.update_yaxes(title_text="Densidad")
    return estilo(fig, "Distribución de datos", 310)


def _graf_atributos(datos):
    y = (datos["x"] / datos["n"]).to_numpy(float) if (datos["n"] != 1).any() else datos["x"].to_numpy(float)
    fig = go.Figure(go.Scatter(x=list(range(1, len(y) + 1)), y=y, mode="lines+markers",
        name="Indicador por muestra", line=dict(color=AZUL, width=2)))
    fig.add_hline(y=float(np.mean(y)), line_color=VERDE,
                  annotation_text=f"Promedio = {np.mean(y):.3f}")
    fig.update_xaxes(title_text="Muestra")
    fig.update_yaxes(title_text="Proporción / conteo")
    return estilo(fig, "Resumen de datos por atributos", 310, derecha=100)


def _modulos():
    st.markdown('<div class="section-heading">Módulos principales</div>', unsafe_allow_html=True)
    for fila_idx in range(0, len(MODULOS), 4):
        cols = st.columns(4)
        for c, (icono, nombre, desc, destino) in zip(cols, MODULOS[fila_idx:fila_idx + 4]):
            with c:
                st.markdown(f'<div class="module-card"><div class="ico">{icono}</div>'
                    f'<div class="name">{nombre}</div><div class="desc">{desc}</div></div>',
                    unsafe_allow_html=True)
                st.button("Abrir módulo", key=f"abrir_{destino}", on_click=ir_a,
                          args=(destino,), use_container_width=True)


def mostrar():
    st.markdown(
        '<div class="hero"><div class="hero-copy">'
        '<div class="hero-kicker">VVM ANALYTICS · PROCESS INTELLIGENCE</div>'
        '<h1>Control Estadístico<br>de Procesos</h1>'
        '<p>Analiza la estabilidad, variabilidad y capacidad de tus procesos productivos. '
        'Carga datos, construye gráficos de control y convierte mediciones en información para decidir.</p>'
        '</div><div class="hero-visual">'
        f'<img class="hero-logo" src="{logo_uri()}" alt="Logo VVM Analytics">'
        '</div></div>', unsafe_allow_html=True)

    df = st.session_state.get("df")
    col = st.session_state.get("col_valor")
    attr = st.session_state.get("datos_attr")
    hay_variables = df is not None and col in df.columns
    hay_atributos = attr is not None
    if not hay_variables and not hay_atributos:
        st.markdown('<div class="section-heading">Comienza aquí</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="module-card" style="text-align:left;min-height:0;padding:1.35rem 1.5rem">'
            '<div class="name" style="font-size:1.05rem">Aún no hay análisis registrados</div>'
            '<div class="desc" style="font-size:.88rem;margin:.5rem 0 .9rem">'
            'Cuando cargues información de un proceso, aquí encontrarás un resumen de los datos y '
            'los resultados obtenidos. Para empezar, importa un archivo Excel o CSV.</div></div>',
            unsafe_allow_html=True)
        a, b, c = st.columns([1.3, 1, 1])
        a.button("📥  Cargar datos del proceso", key="inicio_cargar", type="primary",
                 on_click=ir_a, args=("Datos",), use_container_width=True)
        b.button("📈  Ver gráficos por variables", key="inicio_var",
                 on_click=ir_a, args=("Gráficos por Variables",), use_container_width=True)
        c.button("📊  Ver gráficos por atributos", key="inicio_attr",
                 on_click=ir_a, args=("Gráficos por Atributos",), use_container_width=True)
        st.markdown('<div class="section-heading">¿Qué puedes hacer con VVM Analytics?</div>', unsafe_allow_html=True)
        pasos = st.columns(3)
        for c, numero, titulo, desc in zip(pasos, ("01", "02", "03"),
            ("Carga tus datos", "Analiza el proceso", "Interpreta resultados"),
            ("Importa mediciones o conteos desde Excel o CSV.",
             "Genera gráficos de control por variables y atributos.",
             "Evalúa estabilidad, capacidad y desempeño para apoyar mejoras.")):
            c.markdown(f'<div class="module-card"><div class="ico">{numero}</div>'
                       f'<div class="name">{titulo}</div><div class="desc">{desc}</div></div>',
                       unsafe_allow_html=True)
    else:
        st.markdown('<div class="section-heading">Resumen de tus datos</div>', unsafe_allow_html=True)
        metricas = []
        if hay_variables:
            cap = st.session_state.get("capacidad")
            metricas.extend([
                ("Observaciones", f"{len(df):,}", "azul", "Datos de variables"),
                ("Subgrupos", f"{df['_subgrupo'].nunique():,}" if "_subgrupo" in df else "—", "verde", "Agrupaciones"),
                ("Media", f"{df[col].mean():.3f}", "morado", str(col)),
                ("Cpk", f"{cap['Cpk']:.3f}" if isinstance(cap, dict) and cap.get("Cpk") is not None else "Pendiente", "naranja", "Capacidad"),
            ])
        if hay_atributos:
            metricas.extend([
                ("Muestras con atributos", f"{len(attr):,}", "azul", "Datos de atributos"),
                ("Conteo total", f"{attr['x'].sum():,.0f}", "verde", "Defectuosos / defectos"),
            ])
        cols = st.columns(min(4, len(metricas)))
        for i, (titulo, valor, color, sub) in enumerate(metricas):
            cols[i % len(cols)].markdown(kpi(titulo, valor, color, sub), unsafe_allow_html=True)
        st.markdown('<div class="section-heading">Vista general</div>', unsafe_allow_html=True)
        if hay_variables:
            a, b = st.columns(2)
            a.plotly_chart(_graf_medias(df, col), use_container_width=True, key="inicio_medias")
            b.plotly_chart(_graf_hist(df[col].dropna(), col), use_container_width=True, key="inicio_hist")
        if hay_atributos:
            st.plotly_chart(_graf_atributos(attr), use_container_width=True, key="inicio_attr_graf")
        st.caption("Los resultados detallados y los gráficos de control se encuentran en sus módulos correspondientes.")

    _modulos()