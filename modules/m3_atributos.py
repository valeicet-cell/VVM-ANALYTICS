"""Módulo 3. Gráficos de control por atributos: p, np, c y u (Fase I y Fase II)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from modules import m3b_calidad as calidad
from utils.graficos import AZUL, ROJO, VERDE, NARANJA, GRIS, estilo

DESC = {
    "p": "Fracción de unidades no conformes",
    "np": "Número de unidades no conformes",
    "c": "Número de no conformidades (defectos)",
    "u": "No conformidades por unidad",
}
NOMBRE_PAR = {"p": "p̄", "np": "np̄", "c": "c̄", "u": "ū"}


# ---------------------------------------------------------------- cálculos
def recomendar(defectuosas, n_constante):
    """Asistente: devuelve (gráfico recomendado, justificación)."""
    if defectuosas and n_constante:
        return "np", ("Se cuentan unidades defectuosas y n es constante: el np es el más directo "
                      "(el p también es válido).")
    if defectuosas:
        return "p", "Se cuentan unidades defectuosas y n varía: gráfico p con límites variables."
    if n_constante:
        return "c", ("Se cuentan defectos (una unidad puede tener varios) y el área de oportunidad "
                     "es constante: gráfico c.")
    return "u", ("Se cuentan defectos y el área de oportunidad varía: gráfico u (defectos por "
                 "unidad) con límites variables.")


def estimar(tipo, x, n):
    """Parámetro central estimado en Fase I: p̄, c̄ o ū."""
    if tipo == "c":
        return x.mean()
    return x.sum() / n.sum()


def valores(tipo, x, n):
    """Valor que se grafica en cada punto."""
    return x / n if tipo in ("p", "u") else x


def limites(tipo, par, n):
    """Arreglos (LCI, LC, LCS) por punto, dado el parámetro central."""
    n = np.asarray(n, float)
    if tipo == "p":
        lc = np.full(len(n), par)
        s = 3 * np.sqrt(par * (1 - par) / n)
    elif tipo == "np":
        lc = n * par
        s = 3 * np.sqrt(n * par * (1 - par))
    elif tipo == "c":
        lc = np.full(len(n), par)
        s = np.full(len(n), 3 * np.sqrt(par))
    else:  # u
        lc = np.full(len(n), par)
        s = 3 * np.sqrt(par / n)
    lcs = lc + s
    if tipo == "p":
        lcs = np.minimum(lcs, 1.0)
    return np.maximum(0.0, lc - s), lc, lcs


def racha8(y, lc):
    """Posiciones donde hay 8 puntos consecutivos del mismo lado de la línea central."""
    lado = np.sign(y - lc)
    marcados = []
    for i in range(7, len(y)):
        v = lado[i - 7:i + 1]
        if (v > 0).all() or (v < 0).all():
            marcados.append(i)
    return marcados


def fase1(tipo, x, n, etq, max_iter=10):
    """Fase I automática: quita puntos fuera de límites y recalcula hasta estabilizar."""
    excl, registro = [], []
    for it in range(1, max_iter + 1):
        mask = ~np.isin(etq, excl)
        par = estimar(tipo, x[mask], n[mask])
        y = valores(tipo, x[mask], n[mask])
        lci, lc, lcs = limites(tipo, par, n[mask])
        fuera = list(etq[mask][(y < lci) | (y > lcs)])
        registro.append({"Iteración": it, "Parámetro central": round(float(par), 5),
                         "Puntos fuera": ", ".join(map(str, fuera)) or "ninguno"})
        if not fuera or mask.sum() - len(fuera) < 5:
            break
        excl += fuera
    return excl, pd.DataFrame(registro)


def senales_tramo(y, lci, lc, lcs, ini, fin):
    """Detecta únicamente puntos que están fuera de los límites de control."""
    ys = y[ini:fin]
    r1 = {
        ini + int(i)
        for i in np.flatnonzero(
            (ys < lci[ini:fin]) | (ys > lcs[ini:fin])
        )
    }
    return r1, set()


# ---------------------------------------------------------------- gráfico
def graficar_fase(tipo, etq, y, lci, lc, lcs, excl, r1, r2, ini, fin, fase, rango):
    """Gráfico de una fase (tramo [ini, fin) de las muestras)."""
    idx = np.arange(ini, fin)
    pos = idx + 1
    colores, simbolos = [], []
    for i in idx:
        i = int(i)
        if etq[i] in excl:
            colores.append(GRIS); simbolos.append("x")
        elif i in r1:
            colores.append(ROJO); simbolos.append("circle")
        elif i in r2:
            colores.append(NARANJA); simbolos.append("diamond")
        else:
            colores.append(AZUL); simbolos.append("circle")
    fig = go.Figure()
    # Límites como trazas independientes para que siempre se vean en el gráfico.
    fig.add_trace(go.Scatter(
        x=pos, y=lci[ini:fin], mode="lines", name="LCI",
        hovertemplate="Muestra %{x}<br>LCI: %{y:.5f}<extra></extra>",
        connectgaps=True,
        line=dict(color=ROJO, dash="dash", width=2.4, shape="hv"),
    ))
    fig.add_trace(go.Scatter(
        x=pos, y=lcs[ini:fin], mode="lines", name="LCS",
        hovertemplate="Muestra %{x}<br>LCS: %{y:.5f}<extra></extra>",
        connectgaps=True,
        line=dict(color=ROJO, dash="dash", width=2.4, shape="hv"),
        fill="tonexty", fillcolor="rgba(47,111,222,.06)",
    ))
    fig.add_trace(go.Scatter(
        x=pos, y=lc[ini:fin], mode="lines", name="LC",
        hovertemplate="Muestra %{x}<br>LC: %{y:.5f}<extra></extra>",
        connectgaps=True,
        line=dict(color=VERDE, width=2.4, shape="hv"),
    ))
    fig.add_trace(go.Scatter(
        x=pos, y=y[ini:fin], mode="lines+markers", showlegend=False,
        line=dict(color=AZUL, width=1.8),
        marker=dict(color=colores, symbol=simbolos, size=9, line=dict(color="white", width=1)),
        text=[str(etq[int(i)]) for i in idx],
        hovertemplate="Muestra %{text}<br>Valor: %{y:.4f}<extra></extra>"))
    fig.update_yaxes(range=rango, title_text=DESC[tipo])
    fig.update_xaxes(range=[ini + 0.5, fin + 0.5], title_text="Número de muestra")
    return estilo(fig, f"Gráfico {tipo} · Fase {fase}", 430)


# ---------------------------------------------------------------- pantalla
def mostrar_control():
    datos = st.session_state.get("datos_attr")
    if datos is None:
        st.warning("Primero cargue sus conteos en **Datos → Datos de atributos**.")
        return

    # ---- Barra de configuración superior ----
    with st.container(border=True):
        a = st.columns([1.2, 1.2, 1.8, 1, 1.4])
        objeto = a[0].selectbox("¿Qué se cuenta?", ["Unidades defectuosas", "Defectos por unidad"])
        constante = a[1].selectbox("¿n o área constante?", ["Sí", "No"])
        rec, justif = recomendar(objeto.startswith("Unidades"), constante == "Sí")
        tipo = a[2].selectbox("Gráfico a construir", list(DESC), index=list(DESC).index(rec),
                              format_func=lambda t: f"{t} · {DESC[t]}", key=f"tipo_attr_{rec}")
        n_ii_in = int(a[3].number_input("Puntos Fase II", min_value=0, max_value=500, value=0,
                                        step=1, key="attr_nii"))
        modo = a[4].radio("Depuración Fase I", ["Automática", "Manual"], horizontal=True,
                          key="attr_modo")
    st.caption(f"Asistente de selección: se recomienda el gráfico **{rec}**. {justif} "
               f"Datos cargados: {len(datos)} muestras.")

    datos = datos.dropna().reset_index(drop=True)
    if (datos["x"] < 0).any() or (datos["n"] <= 0).any():
        st.error("Hay conteos negativos o tamaños de muestra no positivos.")
        return
    if tipo in ("p", "np") and (datos["x"] > datos["n"]).any():
        st.error("Hay filas con más defectuosos que unidades inspeccionadas. Revise que haya cargado "
                 "la columna de tamaño de muestra en Datos.")
        return
    if tipo == "np" and datos["n"].nunique() > 1:
        st.error("El gráfico np requiere n constante. Use el gráfico p.")
        return
    N = len(datos)
    if N < 10:
        st.error("Se necesitan al menos 10 muestras.")
        return
    x, n, etq = datos["x"].to_numpy(float), datos["n"].to_numpy(float), datos["etq"].to_numpy()

    n_ii = min(n_ii_in, N - 8)
    if n_ii < n_ii_in:
        st.warning(f"Solo se pueden reservar {N - 8} puntos para Fase II; se usarán {n_ii}.")
    n_f1 = N - n_ii

    # ---- Depuración de Fase I ----
    with st.expander("Depuración de Fase I (causas especiales)"):
        if modo.startswith("Auto"):
            excl, registro = fase1(tipo, x[:n_f1], n[:n_f1], etq[:n_f1])
            st.write("Iteraciones de la Fase I:")
            st.dataframe(registro, hide_index=True)
        else:
            excl = st.multiselect("Puntos a excluir (causa especial identificada)",
                                  list(etq[:n_f1]), key="attr_excl")
        st.caption("Un punto solo se elimina si se identificó una causa especial real; documéntela.")

    # ---- Límites congelados con Fase I depurada ----
    mask1 = ~np.isin(etq[:n_f1], excl)
    par = estimar(tipo, x[:n_f1][mask1], n[:n_f1][mask1])
    y = valores(tipo, x, n)
    lci, lc, lcs = limites(tipo, par, n)

    # Mostrar los límites calculados fuera del gráfico, muestra por muestra.
    st.subheader("Límites de control calculados")
    c_lci, c_lc, c_lcs = st.columns(3)
    c_lci.metric("LCI mínimo", f"{float(np.min(lci)):.5f}")
    c_lc.metric("LC (línea central)", f"{float(np.mean(lc)):.5f}")
    c_lcs.metric("LCS máximo", f"{float(np.max(lcs)):.5f}")

    tabla_limites = pd.DataFrame({
        "Muestra": etq,
        "Fase": ["I" if i < n_f1 else "II" for i in range(N)],
        "LCI": lci,
        "LC": lc,
        "LCS": lcs,
    })
    st.dataframe(
        tabla_limites.style.format({"LCI": "{:.5f}", "LC": "{:.5f}", "LCS": "{:.5f}"}),
        hide_index=True,
        use_container_width=True,
    )
    st.caption(
        "Cuando el tamaño de muestra cambia, los límites LCI y LCS se calculan "
        "para cada muestra; por eso pueden variar entre filas."
    )

    st.metric(f"Parámetro central {NOMBRE_PAR[tipo]} (Fase I depurada)", f"{par:.5f}")
    if tipo in ("p", "np", "u") and (n * par).min() < 5:
        st.warning("Algunas muestras tienen n·p̄ (o n·ū) menor que 5: la aproximación normal "
                   "de los límites puede ser poco precisa.")
    if tipo == "c" and par < 5:
        st.warning("c̄ < 5: la aproximación normal es débil; interprete con cuidado.")

    # ---- Gráficos separados por fase (misma escala vertical) ----
    lo, hi = min(y.min(), lci.min()), max(y.max(), lcs.max())
    pad = 0.08 * (hi - lo) or 1.0
    rango = [lo - pad, hi + pad]
    r1a, r2a = senales_tramo(y, lci, lc, lcs, 0, n_f1)

    st.subheader("Fase I · Establecimiento de los límites")
    st.caption(f"{n_f1} muestras | {len(excl)} excluida(s) | límites calculados con las restantes")
    st.plotly_chart(graficar_fase(tipo, etq, y, lci, lc, lcs, excl, r1a, r2a, 0, n_f1, "I", rango),
                    width="stretch", key="attr_f1")

    r1b, r2b = set(), set()
    st.subheader("Fase II · Monitoreo con límites congelados")
    if n_ii == 0:
        st.info("Para ver la Fase II, indique arriba cuántas muestras reservar (las últimas).")
    else:
        r1b, r2b = senales_tramo(y, lci, lc, lcs, n_f1, N)
        st.caption(f"{n_ii} muestras nuevas evaluadas con los límites de Fase I.")
        st.plotly_chart(graficar_fase(tipo, etq, y, lci, lc, lcs, [], r1b, r2b, n_f1, N, "II", rango),
                        width="stretch", key="attr_f2")

    # ---- Tabla de señales ----
    r1, r2 = r1a | r1b, r2a | r2b
    filas = []
    for i in sorted(r1 | r2):
        reglas = []
        if i in r1:
            reglas.append("Fuera de límites (3σ)")
        if i in r2:
            reglas.append("8 puntos seguidos del mismo lado")
        filas.append({"Muestra": etq[i], "Regla": " + ".join(reglas),
                      "Fase": "II" if i >= n_f1 else "I",
                      "Estado": "Excluido de Fase I" if etq[i] in excl else ""})
    st.subheader("Señales detectadas")
    if filas:
        st.dataframe(pd.DataFrame(filas), hide_index=True)
    else:
        st.success("No se detectaron señales de falta de control.")
    alarmas_ii = [f for f in filas if f["Fase"] == "II"]
    if n_ii > 0:
        if alarmas_ii:
            st.error(f"Fase II: {len(alarmas_ii)} señal(es). Con los límites congelados de Fase I "
                     "hay evidencia de un cambio en el proceso. Investigar causa asignable.")
        else:
            st.success("Fase II: sin señales. El proceso se mantiene estable.")

    st.subheader("Conclusión del análisis de atributos")
    if filas:
        st.warning(f"Se identificaron {len(filas)} señal(es) de posible falta de control, "
                   f"con {len(alarmas_ii)} señal(es) en Fase II. Es necesario investigar las condiciones "
                   "asociadas a las muestras señaladas antes de concluir que el proceso es estable.")
    else:
        st.success("No se detectaron señales con las reglas aplicadas. El comportamiento de los atributos es compatible con control estadístico durante el periodo observado; debe mantenerse el seguimiento.")

    tabla = pd.DataFrame({"Muestra": etq, "Conteo": x, "n": n, "Valor": y.round(5),
                          "LCI": lci.round(5), "LC": lc.round(5), "LCS": lcs.round(5),
                          "Fase": ["I" if i < n_f1 else "II" for i in range(N)]})
    st.session_state["atributos"] = {
        "tipo": tipo, "par": float(par), "excluidos": list(excl), "n_puntos": N,
        "n_f1": n_f1, "n_senales": len(filas), "alarmas_ii": len(alarmas_ii), "tabla": tabla}


def mostrar():
    st.header("Gráficos de control por atributos")
    t1, t2, t3 = st.tabs(["Gráficos de control (p, np, c, u)", "Diagrama de Pareto",
                          "Diagrama de Ishikawa"])
    with t1:
        mostrar_control()
    with t2:
        calidad.mostrar_pareto()
    with t3:
        calidad.mostrar_ishikawa()
