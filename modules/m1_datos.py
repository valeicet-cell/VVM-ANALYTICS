"""Módulo 1. Gestión de datos: variables y atributos, limpieza, subgrupos, resumen y boxplot."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from utils.graficos import AZUL, ROJO, estilo

SIN_N = "(no aplica)"
DERIVADOS = ["limites", "supuestos", "capacidad", "arl_params", "arl_cfg", "arl_comp"]


# ---------------------------------------------------------------- cálculos
def cargar_archivo(archivo):
    """Lee un archivo CSV o Excel subido por el usuario y devuelve un DataFrame."""
    nombre = archivo.name.lower()
    if nombre.endswith(".csv"):
        return pd.read_csv(archivo, sep=None, engine="python")  # detecta coma o punto y coma
    if nombre.endswith((".xlsx", ".xls")):
        return pd.read_excel(archivo)
    raise ValueError("Formato no soportado. Use CSV o Excel.")


def resumen_por_subgrupo(df, col_valor):
    """Tabla con n, media, desviación, mínimo, máximo y rango de cada subgrupo."""
    grupos = df.groupby("_subgrupo")[col_valor]
    tabla = grupos.agg(n="count", media="mean", desv_est="std", minimo="min", maximo="max")
    tabla["rango"] = tabla["maximo"] - tabla["minimo"]
    return tabla.round(4)


def resumen_estadistico(serie):
    """Tabla (Estadístico, Valor) con las medidas descriptivas principales."""
    x = pd.Series(serie).dropna().astype(float)
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    cv = x.std() / x.mean() * 100 if x.mean() != 0 else np.nan
    filas = [("n", len(x)), ("Media", x.mean()), ("Mediana", x.median()),
             ("Desviación estándar", x.std()), ("Varianza", x.var()), ("Mínimo", x.min()),
             ("Primer cuartil (Q1)", q1), ("Tercer cuartil (Q3)", q3), ("Máximo", x.max()),
             ("Rango", x.max() - x.min()), ("Rango intercuartílico", q3 - q1),
             ("Coeficiente de variación (%)", cv), ("Asimetría", x.skew()),
             ("Curtosis (exceso)", x.kurt())]
    return pd.DataFrame([(a, round(float(b), 4)) for a, b in filas], columns=["Estadístico", "Valor"])


def _idx(opciones, claves, exactos=()):
    """Índice de la primera opción cuyo nombre sugiere el contenido buscado."""
    for i, c in enumerate(opciones):
        nombre = str(c).lower()
        if nombre in exactos or any(k in nombre for k in claves):
            return i
    return 0


def _cambio(clave, firma):
    """Si cambian los datos cargados, borra los resultados derivados que quedaron viejos."""
    if st.session_state.get(clave) != firma:
        st.session_state[clave] = firma
        for k in DERIVADOS:
            st.session_state.pop(k, None)


# ---------------------------------------------------------------- gráficos
def graf_boxplot(x, nombre):
    """Diagrama de caja con todos los puntos; los atípicos se marcan en rojo."""
    fig = go.Figure(go.Box(
        y=np.asarray(x, float), name=nombre, boxpoints="all", jitter=0.4, pointpos=-1.7,
        boxmean=True, line=dict(color=AZUL, width=2), fillcolor="rgba(47,111,222,.15)",
        marker=dict(color=AZUL, size=5, opacity=0.6, outliercolor=ROJO)))
    fig.update_yaxes(title_text=nombre)
    return estilo(fig, "Diagrama de caja (boxplot)", 360)


def graf_boxplot_subgrupos(df, col):
    """Un diagrama de caja por subgrupo."""
    etq = df["_subgrupo"].astype(str)
    fig = go.Figure(go.Box(
        x=etq, y=df[col], name="Datos del subgrupo", boxpoints="outliers", boxmean=True,
        line=dict(color=AZUL, width=1.5), fillcolor="rgba(47,111,222,.15)",
        marker=dict(color=ROJO, size=6)))
    fig.update_xaxes(title_text="Subgrupo", categoryorder="array", categoryarray=list(dict.fromkeys(etq)))
    fig.update_yaxes(title_text=col)
    return estilo(fig, "Diagrama de caja por subgrupo", 360)


def mostrar_resumen(serie, nombre, clave, df_sub=None):
    """Resumen estadístico a la izquierda y boxplot a la derecha."""
    st.subheader("Resumen estadístico")
    tabla = resumen_estadistico(serie)
    st.session_state["resumen_est" if clave == "var" else "resumen_est_attr"] = tabla
    vis = tabla.copy()
    vis["Valor"] = [f"{int(v)}" if e == "n" else f"{v:,.4f}" for e, v in zip(tabla["Estadístico"], tabla["Valor"])]
    c = st.columns([1, 1.6])
    c[0].dataframe(vis, hide_index=True, height=35 * (len(vis) + 1) + 3)
    c[1].plotly_chart(graf_boxplot(serie, nombre), width="stretch", key=f"box_{clave}")
    st.caption("Boxplot: la caja va de Q1 a Q3 y la línea interior es la mediana; la línea discontinua es "
               "la media; los bigotes llegan hasta 1,5 veces el rango intercuartílico y los puntos rojos "
               "son valores atípicos.")
    if df_sub is not None and df_sub["_subgrupo"].nunique() <= 40:
        st.plotly_chart(graf_boxplot_subgrupos(df_sub, nombre), width="stretch", key=f"box_sub_{clave}")


# ---------------------------------------------------------------- variables
def tab_variables():
    with st.container(border=True):
        archivo = st.file_uploader("Archivo con las mediciones (CSV o Excel)",
                                   type=["csv", "xlsx", "xls"], key="up_var")
    if archivo is None:
        df0 = st.session_state.get("df")
        if df0 is None:
            st.info("Suba un archivo con una fila por medición, por ejemplo "
                    "datos_prueba/datos_prueba.csv (columnas Subgrupo y Peso_g).")
            return
        col0 = st.session_state["col_valor"]
        st.success(f"Datos de variables cargados: variable «{col0}», {len(df0)} observaciones y "
                   f"{df0['_subgrupo'].nunique()} subgrupos. Para reemplazarlos, suba otro archivo.")
        mostrar_resumen(df0[col0], col0, "var", df0)
        return

    df = cargar_archivo(archivo).copy()
    num = df.select_dtypes(include="number").columns.tolist()
    if not num:
        st.error("El archivo no tiene columnas numéricas.")
        return
    with st.expander("Vista previa del archivo", expanded=False):
        st.dataframe(df.head(20))
        st.caption(f"{df.shape[0]} filas y {df.shape[1]} columnas")

    with st.container(border=True):
        c = st.columns([1.3, 1.2, 1.3, 1.2])
        col_valor = c[0].selectbox("Variable de calidad", num, key="m1_var")
        modo = c[1].radio("Subgrupos", ["Por columna", "Por tamaño fijo"], key="m1_modo")
        if modo == "Por columna":
            cols = list(df.columns)
            col_sub = c[2].selectbox("Columna del subgrupo", cols, key="m1_sub",
                                     index=_idx(cols, ("subgrupo", "lote", "muestra")))
        else:
            tam = int(c[2].number_input("Tamaño del subgrupo", 1, 25, 5, 1, key="m1_n"))
        faltantes = int(df[col_valor].isna().sum())
        quitar = c[3].checkbox(f"Eliminar {faltantes} faltante(s)", value=faltantes > 0,
                               disabled=faltantes == 0, key="m1_na")
    if quitar and faltantes:
        df = df.dropna(subset=[col_valor])
    df = df.reset_index(drop=True)
    if modo == "Por columna":
        df["_subgrupo"] = df[col_sub]
    else:
        df["_subgrupo"] = np.arange(len(df)) // tam + 1
    st.caption("No se eliminan duplicados automáticamente: en un proceso, dos mediciones idénticas "
               "pueden ser perfectamente válidas.")

    with st.expander("Tabla resumen por subgrupo"):
        st.dataframe(resumen_por_subgrupo(df, col_valor))

    _cambio("firma_var", (archivo.name, len(df), col_valor, modo))
    st.session_state["df"] = df
    st.session_state["col_valor"] = col_valor
    st.success(f"Datos de variables listos: {len(df)} observaciones en {df['_subgrupo'].nunique()} "
               "subgrupos. Ya pueden usarse en los demás módulos.")
    mostrar_resumen(df[col_valor], col_valor, "var", df)


# ---------------------------------------------------------------- atributos
def tab_atributos():
    with st.container(border=True):
        archivo = st.file_uploader("Archivo con los conteos por muestra o lote (CSV o Excel)",
                                   type=["csv", "xlsx", "xls"], key="up_attr_m1")

    def resumen(d):
        usa_n = bool((d["n"] != 1).any())
        serie = d["x"] / d["n"] if usa_n else d["x"]
        nombre = "Conteo / tamaño de muestra" if usa_n else "Conteo"
        mostrar_resumen(serie, nombre, "attr")

    if archivo is None:
        d0 = st.session_state.get("datos_attr")
        if d0 is None:
            st.info("Suba un archivo con una fila por muestra, por ejemplo "
                    "datos_prueba/atributos_p.csv (columnas Lote, n y Defectuosos).")
            return
        st.success(f"Datos de atributos cargados: {len(d0)} muestras. Para reemplazarlos, suba otro archivo.")
        resumen(d0)
        return

    df = cargar_archivo(archivo)
    cols = list(df.columns)
    num = df.select_dtypes(include="number").columns.tolist()
    if not num:
        st.error("El archivo no tiene columnas numéricas.")
        return
    with st.expander("Vista previa del archivo"):
        st.dataframe(df.head(20))
    with st.container(border=True):
        c = st.columns(3)
        e_col = c[0].selectbox("Etiqueta de la muestra (lote, fecha)", cols, key="m1a_e")
        x_col = c[1].selectbox("Conteo (defectuosos o defectos)", num, key="m1a_x",
                               index=_idx(num, ("defect",)))
        opciones_n = [SIN_N] + num
        n_col = c[2].selectbox("Tamaño de muestra o unidades (opcional)", opciones_n, key="m1a_n",
                               index=_idx(opciones_n, ("tama", "unidades", "inspec"), ("n",)))
    d = pd.DataFrame({"etq": df[e_col].astype(str), "x": df[x_col]})
    d["n"] = df[n_col] if n_col != SIN_N else 1
    d = d.dropna().reset_index(drop=True)
    if (d["x"] < 0).any() or (d["n"] <= 0).any():
        st.error("Hay conteos negativos o tamaños de muestra no positivos.")
        return
    _cambio("firma_attr", (archivo.name, len(d), x_col, n_col))
    st.session_state["datos_attr"] = d
    st.success(f"Datos de atributos listos: {len(d)} muestras. Ya pueden usarse en "
               "**Gráficos de atributos**.")
    resumen(d)


# ---------------------------------------------------------------- proceso completo
def tab_proceso_completo():
    """Carga un único archivo y distribuye sus columnas a variables y/o atributos."""
    st.markdown("Sube una sola base de datos del proceso. Puedes configurar en el mismo archivo "
                "las mediciones de calidad y los conteos de inspección.")
    with st.container(border=True):
        archivo = st.file_uploader("Archivo completo del proceso (CSV o Excel)",
                                   type=["csv", "xlsx", "xls"], key="up_proceso_completo")
    if archivo is None:
        st.info("El archivo debe contener una fila por observación o muestra. Luego podrás indicar "
                "qué columnas corresponden a mediciones, subgrupos, unidades inspeccionadas y conteos.")
        st.caption("Ejemplo de columnas: Lote, Peso, Unidades_inspeccionadas, Unidades_defectuosas o Numero_defectos.")
        return

    try:
        df = cargar_archivo(archivo).copy()
    except Exception as exc:
        st.error(f"No fue posible leer el archivo: {exc}")
        return
    if df.empty:
        st.error("El archivo no contiene registros.")
        return
    cols = list(df.columns)
    num = df.select_dtypes(include="number").columns.tolist()
    with st.expander("Vista previa de la base de datos", expanded=True):
        st.dataframe(df.head(15), use_container_width=True)
        st.caption(f"{len(df):,} filas · {len(cols)} columnas")

    st.subheader("1. Configurar datos por variables")
    opciones_var = ["(No cargar variables)"] + num
    c = st.columns([1.2, 1.2, 1.2])
    var_col = c[0].selectbox("Variable de calidad", opciones_var, key="pc_var",
                            help="Selecciona una medición numérica, por ejemplo peso, longitud o diámetro.")
    modo = c[1].selectbox("Organización de subgrupos", ["Por columna", "Por tamaño fijo"],
                          key="pc_modo", disabled=var_col == "(No cargar variables)")
    sub_col = None
    tam = 1
    if modo == "Por columna":
        sub_col = c[2].selectbox("Columna de subgrupo", cols, key="pc_sub",
                                 index=_idx(cols, ("subgrupo", "lote", "muestra")),
                                 disabled=var_col == "(No cargar variables)")
    else:
        tam = int(c[2].number_input("Tamaño del subgrupo", min_value=1, max_value=100, value=5,
                                    key="pc_tam", disabled=var_col == "(No cargar variables)"))

    st.subheader("2. Configurar datos por atributos")
    opciones_x = ["(No cargar atributos)"] + num
    a = st.columns([1.2, 1.2, 1.2])
    x_col = a[0].selectbox("Columna de conteo", opciones_x, key="pc_x",
                          help="Puede ser unidades defectuosas (p/np) o número de defectos (c/u).")
    opciones_n = [SIN_N] + num
    n_col = a[1].selectbox("Tamaño de muestra / unidades inspeccionadas", opciones_n, key="pc_n",
                           index=_idx(opciones_n, ("tama", "unidades", "inspec"), ("n",)),
                           disabled=x_col == "(No cargar atributos)")
    etq_col = a[2].selectbox("Etiqueta de muestra", cols, key="pc_etq",
                             index=_idx(cols, ("lote", "muestra", "fecha", "turno")),
                             disabled=x_col == "(No cargar atributos)")

    if st.button("Procesar archivo completo", type="primary", key="procesar_completo", use_container_width=True):
        cargados = []
        if var_col != "(No cargar variables)":
            dvar = df[[var_col]].copy()
            dvar[var_col] = pd.to_numeric(dvar[var_col], errors="coerce")
            dvar = dvar.dropna(subset=[var_col]).copy()
            if dvar.empty:
                st.error("La columna seleccionada para variables no contiene mediciones numéricas válidas.")
                return
            if modo == "Por columna":
                dvar["_subgrupo"] = df.loc[dvar.index, sub_col].astype(str).values
            else:
                dvar["_subgrupo"] = np.arange(len(dvar)) // tam + 1
            dvar = dvar.reset_index(drop=True)
            if dvar["_subgrupo"].nunique() < 2:
                st.error("Se requieren al menos dos subgrupos para continuar con el análisis por variables.")
                return
            _cambio("firma_var", (archivo.name, len(dvar), var_col, modo, str(sub_col), tam))
            st.session_state["df"] = dvar
            st.session_state["col_valor"] = var_col
            cargados.append(f"variables: {len(dvar)} mediciones")
        else:
            st.session_state.pop("df", None)
            st.session_state.pop("col_valor", None)

        if x_col != "(No cargar atributos)":
            d = pd.DataFrame({"etq": df[etq_col].astype(str),
                              "x": pd.to_numeric(df[x_col], errors="coerce")})
            d["n"] = (pd.to_numeric(df[n_col], errors="coerce")
                      if n_col != SIN_N else 1)
            d = d.dropna().reset_index(drop=True)
            if d.empty or (d["x"] < 0).any() or (d["n"] <= 0).any():
                st.error("Los conteos deben ser numéricos y no negativos; el tamaño de muestra debe ser positivo.")
                return
            if n_col != SIN_N and (d["x"] > d["n"]).any():
                st.warning("Hay conteos mayores que el tamaño de muestra. Esto no es válido para gráficos p o np; verifica si seleccionaste unidades defectuosas o número de defectos.")
            _cambio("firma_attr", (archivo.name, len(d), x_col, n_col, etq_col))
            st.session_state["datos_attr"] = d
            cargados.append(f"atributos: {len(d)} muestras")
        else:
            st.session_state.pop("datos_attr", None)

        if cargados:
            st.session_state["origen_datos"] = f"Archivo completo: {archivo.name}"
            st.success("Archivo procesado correctamente. Se cargaron " + " y ".join(cargados) + ".")
            st.info("Ahora puedes consultar los gráficos por variables y/o atributos desde el menú lateral.")
        else:
            st.warning("Selecciona al menos una columna para variables o atributos.")


# ---------------------------------------------------------------- pantalla
def mostrar():
    st.header("Gestión de datos")
    st.caption("Cargue aquí los datos del proceso. Una vez cargados, los demás módulos los usan "
               "automáticamente.")
    t1, t2, t3 = st.tabs(["📁 Proceso completo", "📏 Solo variables", "🔢 Solo atributos"])
    with t1:
        tab_proceso_completo()
    with t2:
        tab_variables()
    with t3:
        tab_atributos()