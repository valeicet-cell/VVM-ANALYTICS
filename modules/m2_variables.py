"""Módulo 2. Gráficos de control por variables: X̄-R, X̄-S e I-MR (Fase I y Fase II)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats

from utils import constantes as const
from utils.arl import arl_shewhart
from utils.graficos import AZUL, ROJO, VERDE, NARANJA, GRIS, estilo

NOMBRES = {
    "X̄-R": ("X̄ (medias)", "R (rangos)"),
    "X̄-S": ("X̄ (medias)", "S (desv. estándar)"),
    "I-MR": ("I (individuales)", "MR (rango móvil)"),
}

# Reglas de zonas: (código, descripción, ventana, umbral en sigmas, mínimo de puntos)
REGLAS = [
    ("R2", "2 de 3 puntos más allá de 2σ (mismo lado)", 3, 2, 2),
    ("R3", "4 de 5 puntos más allá de 1σ (mismo lado)", 5, 1, 4),
    ("R4", "8 puntos consecutivos del mismo lado de la línea central", 8, 0, 8),
]
DESCRIPCION = {"R1": "Punto fuera de los límites de control (3σ)"}
DESCRIPCION.update({c: d for c, d, *_ in REGLAS})
UNIDADES = ["horas", "minutos", "días"]


# ---------------------------------------------------------------- cálculos
def construir_series(df, col_valor, tipo):
    """Devuelve (serie principal, serie de dispersión, tamaño de subgrupo n)."""
    if tipo == "I-MR":
        x = df[col_valor].reset_index(drop=True)
        x.index = x.index + 1
        return x, x.diff().abs(), 1

    g = df.groupby("_subgrupo")[col_valor]
    conteo = g.count()
    n = int(conteo.mode().iloc[0])
    validos = conteo[conteo == n].index  # solo subgrupos completos
    disp = (g.max() - g.min()) if tipo == "X̄-R" else g.std()
    return g.mean().loc[validos], disp.loc[validos], n


def calcular_limites(tipo, principal, disp, n):
    """Límites de control (LCI, LC, LCS) de ambos gráficos y sigma estimada."""
    cl_p = principal.mean()
    cl_d = disp.mean()  # mean() ignora el NaN inicial del rango móvil
    if tipo == "X̄-R":
        sigma = cl_d / const.d2(n)
        a = 3 * sigma / np.sqrt(n)
        lim_d = (const.D3(n) * cl_d, cl_d, const.D4(n) * cl_d)
    elif tipo == "X̄-S":
        sigma = cl_d / const.c4(n)
        a = 3 * sigma / np.sqrt(n)
        w = 3 * cl_d * np.sqrt(1 - const.c4(n) ** 2) / const.c4(n)
        lim_d = (max(0.0, cl_d - w), cl_d, cl_d + w)
    else:  # I-MR
        sigma = cl_d / const.d2(2)
        a = 3 * sigma
        lim_d = (const.D3(2) * cl_d, cl_d, const.D4(2) * cl_d)
    return {"principal": (cl_p - a, cl_p, cl_p + a), "disp": lim_d, "sigma": sigma}


def fuera_de_limites(principal, disp, lim):
    """Etiquetas de los puntos fuera de límites en cualquiera de los dos gráficos."""
    lcl, _, ucl = lim["principal"]
    lcl_d, _, ucl_d = lim["disp"]
    malos = set(principal.index[(principal < lcl) | (principal > ucl)])
    malos |= set(disp.index[(disp < lcl_d) | (disp > ucl_d)])
    return [i for i in principal.index if i in malos]


def fase1_iterativa(principal, disp, tipo, n, max_iter=10):
    """Fase I automática: quita puntos fuera de límites y recalcula hasta estabilizar."""
    excluidos, registro = [], []
    for it in range(1, max_iter + 1):
        mask = ~principal.index.isin(excluidos)
        lim = calcular_limites(tipo, principal[mask], disp[mask], n)
        fuera = fuera_de_limites(principal[mask], disp[mask], lim)
        registro.append({
            "Iteración": it,
            "LCI": round(lim["principal"][0], 4),
            "LC": round(lim["principal"][1], 4),
            "LCS": round(lim["principal"][2], 4),
            "Puntos fuera": ", ".join(map(str, fuera)) or "ninguno",
        })
        if not fuera or mask.sum() - len(fuera) < 3:
            break
        excluidos += fuera
    return excluidos, pd.DataFrame(registro)


def evaluar_senales(serie, lim, con_zonas):
    """Detecta la regla 1 y, si con_zonas=True, las reglas de zonas (Western Electric)."""
    lcl, cl, ucl = lim
    y = serie.values.astype(float)
    etq = list(serie.index)
    out = {"R1": [etq[i] for i in range(len(y)) if y[i] > ucl or y[i] < lcl]}
    sigma = (ucl - cl) / 3
    if con_zonas and sigma > 0:
        z = (y - cl) / sigma
        for codigo, _, w, umbral, minimo in REGLAS:
            marcados = []
            for i in range(w - 1, len(z)):
                v = z[i - w + 1:i + 1]
                if (v > umbral).sum() >= minimo or (v < -umbral).sum() >= minimo:
                    marcados.append(etq[i])  # se marca el último punto de la ventana
            out[codigo] = marcados
    return out


def senales_fase(p, d, lim):
    """Señales de ambos gráficos dentro de un tramo (Fase I o Fase II)."""
    return {"principal": evaluar_senales(p, lim["principal"], con_zonas=True),
            "disp": evaluar_senales(d, lim["disp"], con_zonas=False)}


def rangos_y(principal, disp, lim):
    """Rango vertical común a Fase I y Fase II, para poder compararlas."""
    out = []
    for serie, (lcl, _, ucl) in ((principal, lim["principal"]), (disp, lim["disp"])):
        lo, hi = min(serie.min(), lcl), max(serie.max(), ucl)
        pad = 0.08 * (hi - lo) or 1.0
        out.append([lo - pad, hi + pad])
    return out


# ---------------------------------------------------------------- gráfico
def graficar_fase(principal, disp, lim, sen, excluidos, tipo, fase, offset, rangos):
    """Gráfico de una sola fase: gráfico principal arriba y de dispersión abajo."""
    nom_p, nom_d = NOMBRES[tipo]
    titulos = (f"Fase {fase} · Gráfico {nom_p}", f"Fase {fase} · Gráfico {nom_d}")
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.13,
                        subplot_titles=titulos)
    pos = np.arange(offset + 1, offset + len(principal) + 1)
    series = [(principal, lim["principal"], sen["principal"]),
              (disp, lim["disp"], sen["disp"])]

    for fila, (serie, (lcl, cl, ucl), s) in enumerate(series, start=1):
        if fila == 1:  # zonas A, B y C sombreadas
            sg = (ucl - cl) / 3
            verde, ambar, rojo = "rgba(43,174,102,.08)", "rgba(245,158,11,.10)", "rgba(229,72,77,.09)"
            for y0, y1, color in ((cl - sg, cl + sg, verde), (cl + sg, cl + 2 * sg, ambar),
                                  (cl - 2 * sg, cl - sg, ambar), (cl + 2 * sg, ucl, rojo),
                                  (lcl, cl - 2 * sg, rojo)):
                fig.add_hrect(y0=y0, y1=y1, fillcolor=color, line_width=0, layer="below",
                              row=1, col=1)
        colores, simbolos = [], []
        for etq in serie.index:
            if etq in excluidos:
                colores.append(GRIS); simbolos.append("x")
            elif etq in s.get("R1", []):
                colores.append(ROJO); simbolos.append("circle")
            elif any(etq in v for c, v in s.items() if c != "R1"):
                colores.append(NARANJA); simbolos.append("diamond")
            else:
                colores.append(AZUL); simbolos.append("circle")
        fig.add_trace(go.Scatter(
            x=pos, y=serie.values, mode="lines+markers",
            line=dict(color=AZUL, width=1.5),
            marker=dict(color=colores, symbol=simbolos, size=9, line=dict(color="white", width=1)),
            text=[str(e) for e in serie.index],
            hovertemplate="Subgrupo %{text}<br>Valor: %{y:.4f}<extra></extra>",
            showlegend=False), row=fila, col=1)
        for valor, nombre, color, dash in ((ucl, "LCS", ROJO, "dash"), (cl, "LC", VERDE, "solid"),
                                           (lcl, "LCI", ROJO, "dash")):
            fig.add_hline(y=valor, row=fila, col=1, line_dash=dash, line_color=color, line_width=1.5,
                          annotation_text=f"{nombre}={valor:.3f}", annotation_position="top right")
        fig.update_yaxes(range=rangos[fila - 1], row=fila, col=1)

    fig.update_xaxes(range=[offset + 0.5, offset + len(principal) + 0.5])
    fig.update_xaxes(title_text="Subgrupo / observación", row=2, col=1)
    fig = estilo(fig, None, 620, derecha=95)

    # Títulos de cada subgráfico: se suben para que no toquen el marco del gráfico
    fig.update_layout(margin=dict(t=80))
    for a in fig.layout.annotations:
        if a.text in titulos:
            a.update(yshift=18, font=dict(size=15, color="#0F2740"))
    return fig


# ---------------------------------------------------------------- pantalla
def mostrar():
    st.header("Módulo 2. Gráficos de control por variables")
    df = st.session_state.get("df")
    col = st.session_state.get("col_valor")
    if df is None:
        st.warning("Primero cargue datos en **Gestión de datos** (Módulo 1).")
        return

    cfg = st.session_state.get("arl_cfg", {})
    st.session_state.setdefault("m2_h", float(cfg.get("h", 1.0)))
    st.session_state.setdefault("m2_u", cfg.get("unidad", "horas"))
    st.session_state.setdefault("m2_d", float(cfg.get("delta", 1.0)))

    # ---- Barra de configuración superior ----
    with st.container(border=True):
        c = st.columns([1.1, 1, 1.5, 0.9, 0.9, 0.9])
        tipo = c[0].selectbox("Tipo de gráfico", list(NOMBRES), key="m2_tipo")
        principal, disp, n = construir_series(df, col, tipo)
        if tipo != "I-MR" and not 2 <= n <= const.N_MAX:
            st.error(f"El tamaño de subgrupo es {n}; para {tipo} debe estar entre 2 y {const.N_MAX}. "
                     "Revise los subgrupos en el Módulo 1 o use I-MR.")
            return
        if len(principal) < 8:
            st.error("Se necesitan al menos 8 puntos para construir el gráfico.")
            return
        n_ii = int(c[1].number_input("Puntos Fase II", min_value=0, max_value=len(principal) - 5,
                                     value=0, step=1, help="Los últimos puntos se reservan para Fase II."))
        modo = c[2].radio("Depuración Fase I", ["Automática", "Manual"], horizontal=True, key="m2_modo")
        h = c[3].number_input("Intervalo h", min_value=0.01, max_value=1000.0, step=0.25, key="m2_h",
                              help="Tiempo entre muestras (para el ATS).")
        unidad = c[4].selectbox("Unidad", UNIDADES, key="m2_u")
        delta = c[5].number_input("δ a detectar (σ)", min_value=0.05, max_value=5.0, step=0.05, key="m2_d",
                                  help="Desplazamiento de la media que se quiere detectar (para el ARL1).")

    st.caption(f"Variable: **{col}** | Tamaño de subgrupo n = {n} | Puntos: {len(principal)}")
    if len(principal) < 20:
        st.warning("Se recomiendan al menos 20-25 subgrupos en Fase I para límites confiables.")
    n_f1 = len(principal) - n_ii
    p1, d1 = principal.iloc[:n_f1], disp.iloc[:n_f1]

    # ---- Depuración de Fase I ----
    with st.expander("Depuración de Fase I y registro de causas especiales"):
        if modo.startswith("Auto"):
            excluidos, registro = fase1_iterativa(p1, d1, tipo, n)
            st.write("Iteraciones de la Fase I:")
            st.dataframe(registro, hide_index=True)
        else:
            excluidos = st.multiselect("Puntos a excluir (causa especial identificada)",
                                       list(p1.index), key=f"excl_{tipo}")
        st.caption("Un punto solo debe eliminarse si se identificó una causa especial real. "
                   "Documéntela; no se elimina un punto solo por salirse.")
        st.text_area("Registro de causas especiales (para el informe)", key=f"causas_{tipo}",
                     placeholder="Ej.: Subgrupo 12 - cambio de lote de materia prima...")

    # ---- Límites de Fase I depurada (se congelan para Fase II) ----
    mask = ~p1.index.isin(excluidos)
    lim = calcular_limites(tipo, p1[mask], d1[mask], n)
    c = st.columns(4)
    c[0].metric("LCI", f"{lim['principal'][0]:.4f}")
    c[1].metric("Línea central", f"{lim['principal'][1]:.4f}")
    c[2].metric("LCS", f"{lim['principal'][2]:.4f}")
    c[3].metric("σ estimada (dentro)", f"{lim['sigma']:.4f}")

    # ---- ARL y ATS del gráfico ----
    alfa = 2 * stats.norm.sf(3.0)
    arl0 = 1 / alfa
    arl1 = float(arl_shewhart(delta, n, 3.0))
    with st.container(border=True):
        st.markdown("**Desempeño del gráfico de medias (límites a 3σ)**")
        a = st.columns(5)
        a[0].metric("α (falsa alarma)", f"{alfa:.5f}")
        a[1].metric("ARL0", f"{arl0:.1f} muestras")
        a[2].metric("ATS0", f"{arl0 * h:.1f} {unidad}")
        a[3].metric(f"ARL1 (δ = {delta:g}σ)", f"{arl1:.2f} muestras")
        a[4].metric("ATS1", f"{arl1 * h:.2f} {unidad}")
        st.caption("Calculados con n = {} para el gráfico {}, bajo normalidad e independencia y tomando σ "
                   "estimada como verdadera. Si además usa reglas de zonas, el ARL0 real es menor. Para "
                   "diseñar otros valores use el Módulo 6.".format(n, NOMBRES[tipo][0].split()[0]))

    # ---- Señales por fase (con límites congelados) ----
    rangos = rangos_y(principal, disp, lim)
    sen1 = senales_fase(p1, d1, lim)
    p2, d2s = principal.iloc[n_f1:], disp.iloc[n_f1:]
    sen2 = senales_fase(p2, d2s, lim) if n_ii > 0 else None

    st.subheader("Fase I · Establecimiento de los límites")
    st.caption(f"{n_f1} puntos | {len(excluidos)} excluido(s) | límites calculados con los puntos restantes")
    st.plotly_chart(graficar_fase(p1, d1, lim, sen1, excluidos, tipo, "I", 0, rangos),
                    width="stretch", key="fig_fase1")

    st.subheader("Fase II · Monitoreo con límites congelados")
    if n_ii == 0:
        st.info("Para ver la Fase II, indique arriba cuántos puntos reservar (por ejemplo, los últimos 8). "
                "Esos puntos se evalúan con los límites de la Fase I, sin recalcularlos.")
    else:
        st.caption(f"{n_ii} puntos nuevos evaluados con los límites de Fase I (misma escala vertical).")
        st.plotly_chart(graficar_fase(p2, d2s, lim, sen2, [], tipo, "II", n_f1, rangos),
                        width="stretch", key="fig_fase2")
    st.caption("Rojo: fuera de límites | Naranja: otra regla de zonas | Gris con X: excluido de Fase I | "
               "Bandas: zonas A, B y C")

    # ---- Tabla de señales ----
    pos_de = {e: i for i, e in enumerate(principal.index)}
    filas = []

    def agregar(fase, sen):
        for grafico, nombre in (("principal", NOMBRES[tipo][0]), ("disp", NOMBRES[tipo][1])):
            for codigo, etqs in sen[grafico].items():
                for e in etqs:
                    filas.append({"Gráfico": nombre, "Regla": DESCRIPCION[codigo], "Subgrupo": e,
                                  "Fase": fase, "Orden": pos_de[e],
                                  "Estado": "Excluido de Fase I" if e in excluidos else ""})

    agregar("I", sen1)
    if sen2:
        agregar("II", sen2)
    st.subheader("Señales detectadas")
    if filas:
        tabla = pd.DataFrame(filas).sort_values("Orden").drop(columns="Orden")
        st.dataframe(tabla, hide_index=True)
    else:
        st.success("No se detectaron señales de falta de control.")

    alarmas_ii = [f for f in filas if f["Fase"] == "II"]
    if n_ii > 0:
        if alarmas_ii:
            st.error(f"Fase II: {len(alarmas_ii)} señal(es) de alarma. Con los límites congelados de "
                     "Fase I, el proceso muestra evidencia de un cambio. Investigar causa asignable.")
        else:
            st.success("Fase II: sin señales. El proceso se mantiene estable respecto a la Fase I.")

    st.subheader("Conclusión del análisis de control")
    lcl_p, cl_p, ucl_p = lim["principal"]
    cap = st.session_state.get("capacidad")
    cp_v = cap.get("Cp") if cap else None
    cpk_v = cap.get("Cpk") if cap else None
    if cpk_v is not None:
        cp_txt = "—" if cp_v is None else f"{cp_v:.3f}"
        cap_txt = f"Capacidad del proceso (módulo Capacidad): Cp = {cp_txt} y Cpk = {cpk_v:.3f}."
    else:
        cap_txt = ("Cp y Cpk aún no están disponibles: se calculan en el módulo Capacidad "
                   "una vez ingresadas las especificaciones (LEI y LES).")
    resumen = (f"Fase I con {n_f1} puntos ({len(excluidos)} excluido(s)): LCI = {lcl_p:.4f}, "
               f"LC = {cl_p:.4f}, LCS = {ucl_p:.4f} y σ estimada = {lim['sigma']:.4f}. "
               f"Desempeño del gráfico (límites a 3σ, n = {n}): α = {alfa:.5f}, "
               f"ARL0 = {arl0:.1f} muestras (ATS0 = {arl0 * h:.1f} {unidad}); ante un desplazamiento de "
               f"{delta:g}σ, ARL1 = {arl1:.2f} muestras (ATS1 = {arl1 * h:.2f} {unidad}). {cap_txt}")
    if filas:
        st.warning(f"El análisis identificó {len(filas)} señal(es) de comportamiento no aleatorio, "
                   f"incluidas {len(alarmas_ii)} en Fase II. El proceso requiere investigación de causas "
                   "especiales antes de considerarse estable; las señales deben contrastarse con registros "
                   f"operativos. {resumen}")
    else:
        st.success("No se identificaron señales con las reglas aplicadas. Los datos son compatibles con un "
                   "proceso bajo control estadístico en el periodo evaluado; se recomienda mantener el "
                   f"monitoreo para confirmar la continuidad de este comportamiento. {resumen}")

    # Guardar para los demás módulos
    st.session_state["arl_cfg"] = {"h": h, "unidad": unidad, "delta": delta, "L": 3.0, "n": n}
    st.session_state["limites"] = {
        "tipo": tipo, "n": n, "sigma": lim["sigma"],
        "media": lim["principal"][1], "excluidos": excluidos,
        "lcl": lim["principal"][0], "ucl": lim["principal"][2],
        "n_f1": n_f1, "n_senales": len(filas), "alarmas_ii": len(alarmas_ii),
        "alfa": alfa, "arl0": arl0, "ats0": arl0 * h, "arl1": arl1, "ats1": arl1 * h,
        "delta": delta, "h": h, "unidad": unidad,
    }