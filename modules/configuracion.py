"""Preferencias básicas de presentación y ayuda del sistema."""
import streamlit as st


def mostrar():
    st.title("Configuración")
    st.caption("Preferencias generales de VVM Analytics.")
    with st.container(border=True):
        st.subheader("Información del software")
        st.write("**Nombre:** VVM Analytics")
        st.write("**Versión:** 1.0.0")
        st.write("**Tecnología:** Python y Streamlit")
        st.write("**Módulos de control:** Variables y atributos, disponibles por separado.")
    with st.container(border=True):
        st.subheader("Estado de la sesión")
        df = st.session_state.get("df")
        attr = st.session_state.get("datos_attr")
        st.write(f"Datos por variables: {'Cargados' if df is not None else 'No cargados'}")
        st.write(f"Datos por atributos: {'Cargados' if attr is not None else 'No cargados'}")
        if st.button("Limpiar datos de la sesión", type="secondary"):
            claves = [k for k in st.session_state.keys() if k not in ("pagina",)]
            for clave in claves:
                del st.session_state[clave]
            st.success("Se limpiaron los datos temporales de esta sesión.")
            st.rerun()
    st.info("Los archivos originales y los datos de ejemplo no se modifican desde esta pantalla.")
