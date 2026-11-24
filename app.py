"""Aplicación principal de VVM Analytics."""
import streamlit as st

from modules import (
    inicio, m1_datos, m2_variables, m3_atributos, m4_supuestos,
    m5_capacidad, m6_arl_ats, m7_aceptacion, m8_reportes,
)
from utils.estilos import aplicar_estilos, logo_uri
st.set_page_config(
    page_title="VVM Analytics | Control Estadístico de Procesos",
    page_icon="logo.jpeg",
    layout="wide",
    initial_sidebar_state="expanded",
)
aplicar_estilos()

# Las páginas de variables y atributos se mantienen deliberadamente separadas.
PAGINAS = {
    "Inicio": ("🏠", inicio.mostrar),
    "Datos": ("🗄️", m1_datos.mostrar),
    "Gráficos por Variables": ("📈", m2_variables.mostrar),
    "Gráficos por Atributos": ("📊", m3_atributos.mostrar),
    "Capacidad": ("🎯", m5_capacidad.mostrar),
    "Supuestos": ("✅", m4_supuestos.mostrar),
    "Diseño de Gráficos": ("⚙️", m6_arl_ats.mostrar),
    "Muestreo de Aceptación": ("📦", m7_aceptacion.mostrar),
    "Reportes": ("📄", m8_reportes.mostrar),
    "Configuración": ("⚙️", None),
}

if "pagina" not in st.session_state or st.session_state["pagina"] not in PAGINAS:
    st.session_state["pagina"] = "Inicio"

with st.sidebar:
    st.markdown(
        '<div class="brand-lockup brand-lockup-logo">'
        f'<img class="brand-logo-full" src="{logo_uri()}" alt="VVM Analytics">'
        '</div>',
        unsafe_allow_html=True,
    )
    st.markdown('<div class="sidebar-section-label">MENÚ PRINCIPAL</div>', unsafe_allow_html=True)
    st.radio(
        "Navegación",
        options=list(PAGINAS.keys()),
        format_func=lambda nombre: f"{PAGINAS[nombre][0]}   {nombre}",
        key="pagina",
        label_visibility="collapsed",
    )
    st.markdown(
        '<div class="sidebar-note"><span>✦</span><br>Decisiones basadas<br>en evidencia</div>',
        unsafe_allow_html=True,
    )
    st.caption("VVM Analytics · Versión 1.0.0")

pagina = st.session_state["pagina"]
if pagina == "Configuración":
    from modules.configuracion import mostrar
    mostrar()
else:
    PAGINAS[pagina][1]()
