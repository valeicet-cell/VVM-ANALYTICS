"""Módulo 6. Diseño de gráficos: ARL0, ARL1 y ATS (diseño por metas, datos reales y simulación)."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from modules import m2_variables as m2
from utils.arl import (arl_shewhart, largo_corrida, n_minimo, simular_arl)
from utils.datos import datos_estables, tipo_grafico_por_defecto
from utils.graficos import AZUL, ROJO, VERDE, NARANJA, GRIS, estilo

DELTAS = [0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0]
OPCIONES_N = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 15, 20, 25]
UNIDADES = ["horas", "minutos", "días"]


# ---------------------------------------------------------------- utilidades
def parametros_proceso():
    """(μ0, σ, n de los datos) tomados del Módulo 2 o calculados al vuelo; None si no hay datos."""
    df = st.session_state.get("df")
    col = st.session_state.get("col_valor")
    if df is None:
        return None
    lim = st.session_state.get("limites")
    if lim:
        return float(lim["media"]), float(lim["sigma"]), int(lim["n"])
    try:
        tipo = tipo_grafico_por_defecto(df, col)
        p, d, n = m2.construir_series(df, col, tipo)
        l = m2.calcular_limites(tipo, p, d, n)
        return float(l["principal"][1]), float(l["sigma"]), int(n)
    except Exception:
        return None


def _aplicar(L, n, h):
    """Copia un diseño a la barra de configuración (se ejecuta al pulsar el botón)."""
    st.session_state["m6_L"] = float(L)
    st.session_state["m6_n"] = int(n)
    st.session_state["m6_h"] = float(h)


def _nueva_semilla():
    st.session_state["s_seed"] = int(st.session_state.get("s_seed", 42)) + 1


def grafico_datos(L, n, h, unidad, prm, clave):
    """Gráfico X̄ con límites ±Lσ/√n aplicado a los datos reales (reagrupados en n)."""
    if prm is None:
        st.info("Cargue datos en el Módulo 1 para aplicar el diseño a sus datos.")
        return
    mu0, sigma, n_dat = prm
    df = st.session_state["df"]
    col = st.session_state["col_valor"]
    d = datos_estables(df, True).dropna(subset=[col])
    if n == n_dat and "_subgrupo" in d.columns:
        g = d.groupby("_subgrupo", sort=False)[col]
        y = g.mean()[g.count() == n].to_numpy(float)
    else:
        x = d[col].to_numpy(float)
        k = len(x) // n
        y = x[:k * n].reshape(k, n).mean(axis=1) if k > 0 else np.array([])
    if len(y) < 2:
        st.info("Con este n no hay suficientes muestras en los datos.")
        return
    ancho = L * sigma / np.sqrt(n)
    lcs, lci = mu0 + ancho, mu0 - ancho
    fuera = (y > lcs) | (y < lci)
    pos = np.arange(1, len(y) + 1)
    fig = go.Figure()
    fig.add_hrect(y0=lci, y1=lcs, fillcolor="rgba(91,75,219,.05)", line_width=0)
    fig.add_trace(go.Scatter(
        x=pos, y=y, mode="lines+markers", line=dict(color=AZUL, width=1.5), showlegend=False,
        marker=dict(color=[ROJO if f else AZUL for f in fuera], size=9, line=dict(color="white", width=1)),
        customdata=pos * h,
        hovertemplate="Muestra %{x}<br>X̄: %{y:.4f}<br>Tiempo: %{customdata:.1f} " + unidad + "<extra></extra>"))
    for v, nom, color, dash in ((lcs, "LCS", ROJO, "dash"), (mu0, "LC", VERDE, "solid"), (lci, "LCI", ROJO, "dash")):
        fig.add_hline(y=v, line_color=color, line_dash=dash, annotation_text=f"{nom}={v:.3f}",
                      annotation_position="top right")
    fig.update_xaxes(title_text="Muestra")
    fig.update_yaxes(title_text="Media de la muestra")
    st.plotly_chart(estilo(fig, f"Gráfico X̄ con L = {L:.3f}, n = {n} (μ0 = {mu0:.3f}, σ = {sigma:.3f})",
                           420, derecha=95), width="stretch", key=f"datos_{clave}")
    m = st.columns(4)
    m[0].metric("Muestras", len(y))
    m[1].metric("Señales", int(fuera.sum()))
    m[2].metric("Tasa de señales", f"{fuera.mean() * 100:.1f} %")
    m[3].metric("Tiempo cubierto", f"{len(y) * h:.1f} {unidad}")
    if n != n_dat:
        st.caption(f"Los datos originales usan n = {n_dat}; aquí se reagrupan en orden de captura en "
                   f"muestras de {n}. Úselo como ilustración del diseño.")
    st.caption(f"Con ARL0 = {1 / (2 * stats.norm.sf(L)):.0f} se esperan falsas alarmas en ~"
               f"{100 / (1 / (2 * stats.norm.sf(L))):.2f} % de las muestras si el proceso está estable.")


# ---------------------------------------------------------------- pantalla
def mostrar():
    st.header("Módulo 6. Diseño de gráficos: ARL0, ARL1 y ATS")
    prm = parametros_proceso()
    cfg = st.session_state.get("arl_cfg", {})
    n_ini = int(min(max(cfg.get("n") or (prm[2] if prm else 5), 1), 200))
    for k, v in {"m6_L": float(cfg.get("L", 3.0)), "m6_n": n_ini, "m6_h": float(cfg.get("h", 1.0)),
                 "m6_u": cfg.get("unidad", "horas"), "m6_d": float(cfg.get("delta", 1.0))}.items():
        st.session_state.setdefault(k, v)

    # ---- Barra de configuración superior ----
    with st.container(border=True):
        c = st.columns(5)
        L = c[0].number_input("Amplitud L (σ)", 1.0, 5.0, step=0.05, format="%.3f", key="m6_L")
        n = int(c[1].number_input("Tamaño de muestra n", 1, 200, step=1, key="m6_n"))
        h = c[2].number_input("Intervalo h", 0.01, 1000.0, step=0.25, key="m6_h")
        unidad = c[3].selectbox("Unidad de tiempo", UNIDADES, key="m6_u")
        delta = c[4].number_input("Cambio a detectar δ (σ)", 0.05, 5.0, step=0.05, key="m6_d")

    alfa = 2 * stats.norm.sf(L)
    arl0 = 1 / alfa
    arl1 = float(arl_shewhart(delta, n, L))
    m = st.columns(5)
    m[0].metric("α (falsa alarma)", f"{alfa:.5f}")
    m[1].metric("ARL0", f"{arl0:.1f} muestras")
    m[2].metric("ATS0", f"{arl0 * h:.1f} {unidad}")
    m[3].metric(f"ARL1 (δ = {delta:g}σ)", f"{arl1:.2f} muestras")
    m[4].metric("ATS1", f"{arl1 * h:.2f} {unidad}")
    if prm:
        st.caption(f"Valores iniciales tomados de sus datos (Módulo 2): n = {prm[2]}, μ0 = {prm[0]:.4f}, "
                   f"σ = {prm[1]:.4f}. Cambie los parámetros de arriba para explorar otros diseños.")
    else:
        st.caption("Sin datos cargados: se usan valores por defecto. Cargue datos (Módulo 1) y ejecute el "
                   "Módulo 2 para partir de su proceso.")

    d_tab = np.array(DELTAS)
    a1_tab = arl_shewhart(d_tab, n, L)
    tabla = pd.DataFrame({"Desplazamiento δ (σ)": d_tab, "Prob. de no detectar (β)": (1 - 1 / a1_tab).round(4),
                          "ARL1 (muestras)": a1_tab.round(2), f"ATS1 ({unidad})": (a1_tab * h).round(2)})
    st.session_state["arl_cfg"] = {"h": h, "unidad": unidad, "delta": delta, "L": L, "n": n}
    st.session_state["arl_params"] = {
        "L": L, "n": n, "h": h, "unidad": unidad, "arl0": arl0, "ats0": arl0 * h, "delta": delta,
        "arl1": arl1, "arl1_d1": float(arl_shewhart(1.0, n, L)), "tabla": tabla}

    st.markdown("**Análisis del diseño actual**")
    st.info(f"Con los parámetros seleccionados, el gráfico presenta un ARL₀ de {arl0:.1f} muestras, "
            f"equivalente a un ATS₀ de {arl0*h:.1f} {unidad}. Ante un desplazamiento de {delta:g}σ, "
            f"el tiempo medio de señal esperado es {arl1*h:.2f} {unidad} (ARL₁ = {arl1:.2f}). "
            "La conveniencia del diseño debe juzgarse equilibrando la rapidez de detección, la frecuencia "
            "de falsas alarmas y la carga de inspección.")

    t1, t2, t3, t4, t5 = st.tabs(["🎯 Diseño por metas", "📈 Gráfico con sus datos",
                                      "🧪 Simulación del proceso", "📉 Curvas y tabla",
                                      "n vs. frecuencia"])

    # ---------------- 1. Diseño por metas ----------------
    with t1:
        st.caption("Marque las metas que debe cumplir el gráfico X̄. El software calcula el L y el menor n "
                   "que las cumplen, usando h y δ de la barra superior.")
        c = st.columns(4)
        u0 = c[0].checkbox("Meta de ARL0", True, key="g_u0")
        o0 = c[0].number_input("ARL0 ≥", 10.0, 100000.0, 370.0, 10.0, key="g_o0", disabled=not u0)
        u1 = c[1].checkbox("Meta de ARL1", True, key="g_u1")
        o1 = c[1].number_input("ARL1 ≤ (muestras)", 1.0, 1000.0, 5.0, 0.5, key="g_o1", disabled=not u1)
        u2 = c[2].checkbox("Meta de ATS1", False, key="g_u2")
        o2 = c[2].number_input(f"ATS1 ≤ ({unidad})", 0.01, 10000.0, 4.0, 0.5, key="g_o2", disabled=not u2)
        nmax = int(c[3].number_input("n máximo permitido", 1, 500, 25, 1, key="g_nm"))

        Ld = float(stats.norm.ppf(1 - 1 / (2 * o0))) if u0 else float(L)
        tope = np.inf
        if u1:
            tope = min(tope, o1)
        if u2:
            tope = min(tope, o2 / h)
        nd = n_minimo(delta, Ld, tope, nmax) if np.isfinite(tope) else n
        if nd is None:
            st.error(f"Con L = {Ld:.3f} no se logra la meta con n ≤ {nmax} (δ = {delta:g}σ). Pruebe "
                     "aumentar el n máximo, relajar el ARL1/ATS1, aceptar un ARL0 menor o reducir h.")
        else:
            a0 = 1 / (2 * stats.norm.sf(Ld))
            a1 = float(arl_shewhart(delta, nd, Ld))
            st.success(f"Diseño propuesto: **L = {Ld:.3f}, n = {nd}, h = {h:g} {unidad}**.")
            k = st.columns(5)
            k[0].metric("ARL0", f"{a0:.1f}")
            k[1].metric("ATS0", f"{a0 * h:.1f} {unidad}")
            k[2].metric("ARL1", f"{a1:.2f}")
            k[3].metric("ATS1", f"{a1 * h:.2f} {unidad}")
            k[4].metric("Carga (n/h)", f"{nd / h:.2f} u/{unidad[:-1] if unidad != 'días' else 'día'}")
            if u2:
                st.caption(f"Con n = {nd}, el h máximo para cumplir la meta de ATS1 es {o2 / a1:.2f} {unidad}.")
            st.button("Usar este diseño en la barra de configuración", on_click=_aplicar,
                      args=(Ld, nd, h), key="g_apl")
            filas = []
            for nn in range(1, min(nmax, 12) + 1):
                v = float(arl_shewhart(delta, nn, Ld))
                filas.append({"n": nn, "ARL1": round(v, 2), "ATS1": round(v * h, 2),
                              "¿Cumple?": "Sí" if v <= tope else "No"})
            with st.expander("Ver ARL1 y ATS1 según n (con este L)"):
                st.dataframe(pd.DataFrame(filas), hide_index=True)
            st.markdown("**Así se vería este diseño en sus datos:**")
            grafico_datos(Ld, nd, h, unidad, prm, "diseno")

    # ---------------- 2. Gráfico con los datos reales ----------------
    with t2:
        st.caption("Gráfico X̄ construido con los parámetros de la barra superior (L, n) sobre sus datos.")
        grafico_datos(L, n, h, unidad, prm, "config")

    # ---------------- 3. Simulación del proceso ----------------
    with t3:
        mu0, sigma = (prm[0], prm[1]) if prm else (0.0, 1.0)
        if not prm:
            st.info("Sin datos: se simula un proceso estándar (μ = 0, σ = 1).")
        st.caption(f"Se simula un proceso con μ0 = {mu0:.3f} y σ = {sigma:.3f}. En la muestra elegida la "
                   f"media se desplaza δ = {delta:g}σ (barra superior) y el gráfico usa L = {L:.3f}, n = {n}.")
        st.session_state.setdefault("s_seed", 42)
        c = st.columns(5)
        k_tot = int(c[0].number_input("Muestras a simular", 20, 500, 80, 10, key="s_k"))
        k_sh = min(int(c[1].number_input("El cambio ocurre en la muestra", 2, 500, 40, 1, key="s_c")), k_tot)
        fac = float(c[2].number_input("Factor de σ tras el cambio", 0.5, 4.0, 1.0, 0.1, key="s_f"))
        c[3].number_input("Semilla", 0, 999999, step=1, key="s_seed")
        c[4].button("🎲 Nueva simulación", on_click=_nueva_semilla)

        rng = np.random.default_rng(int(st.session_state["s_seed"]))
        idx = np.arange(1, k_tot + 1)
        post = idx >= k_sh
        xb = rng.normal(mu0 + np.where(post, delta * sigma, 0.0),
                        np.where(post, fac * sigma, sigma) / np.sqrt(n))
        ancho = L * sigma / np.sqrt(n)
        lcs, lci = mu0 + ancho, mu0 - ancho
        sen = (xb > lcs) | (xb < lci)
        falsas = np.flatnonzero(sen & ~post)
        det = np.flatnonzero(sen & post)
        rl = int(det[0] + 1 - (k_sh - 1)) if len(det) else None

        colores = [ROJO if (sen[i] and post[i]) else NARANJA if sen[i] else AZUL for i in range(k_tot)]
        fig = go.Figure()
        fig.add_vrect(x0=k_sh - 0.5, x1=k_tot + 0.5, fillcolor="rgba(229,72,77,.06)", line_width=0,
                      annotation_text="Proceso desplazado", annotation_position="top left")
        fig.add_trace(go.Scatter(x=idx, y=xb, mode="lines+markers", showlegend=False,
                                 line=dict(color=AZUL, width=1.5),
                                 marker=dict(color=colores, size=8, line=dict(color="white", width=1)),
                                 customdata=idx * h,
                                 hovertemplate="Muestra %{x}<br>X̄: %{y:.4f}<br>Tiempo: %{customdata:.1f} "
                                               + unidad + "<extra></extra>"))
        for v, nom, color, dash in ((lcs, "LCS", ROJO, "dash"), (mu0, "LC", VERDE, "solid"), (lci, "LCI", ROJO, "dash")):
            fig.add_hline(y=v, line_color=color, line_dash=dash, annotation_text=f"{nom}={v:.3f}",
                          annotation_position="top right")
        if rl:
            fig.add_vline(x=int(det[0] + 1), line_dash="dot", line_color=ROJO,
                          annotation_text="Detección", annotation_position="bottom right")
        fig.update_xaxes(title_text="Muestra")
        fig.update_yaxes(title_text="Media de la muestra")
        st.plotly_chart(estilo(fig, "Proceso simulado con cambio", 430, derecha=95), width="stretch",
                        key="g_sim")
        k = st.columns(4)
        k[0].metric("Falsas alarmas antes del cambio", len(falsas),
                    f"esperadas ≈ {(k_sh - 1) / arl0:.2f}", delta_color="off")
        k[1].metric("Muestras hasta detectar", rl if rl else "no detectó",
                    f"teórico (promedio) {arl1:.1f}" if fac == 1 else None, delta_color="off")
        k[2].metric("ATS observado", f"{rl * h:.1f} {unidad}" if rl else "—")
        k[3].metric("ATS teórico", f"{arl1 * h:.1f} {unidad}" if fac == 1 else "—")
        st.caption("Una sola simulación puede salir muy distinta del promedio: el largo de corrida es "
                   "aleatorio. Pulse «Nueva simulación» varias veces o repita el experimento abajo.")

        with st.expander("Repetir el experimento muchas veces (distribución del largo de corrida)"):
            reps = int(st.number_input("Réplicas", 200, 20000, 2000, 200, key="s_reps"))
            if st.button("Ejecutar réplicas", key="s_run"):
                rl_arr, trunc = largo_corrida("Shewhart", delta * np.sqrt(n), {"L": L}, reps, 5000,
                                              np.random.default_rng(int(st.session_state["s_seed"])), sd=fac)
                tope_x = float(np.percentile(rl_arr, 99))
                fh = go.Figure(go.Histogram(x=np.clip(rl_arr, 0, tope_x), nbinsx=40,
                                            marker=dict(color="rgba(91,75,219,.5)", line=dict(color="white", width=1))))
                fh.add_vline(x=float(rl_arr.mean()), line_color=ROJO, annotation_text="Promedio simulado")
                if fac == 1:
                    fh.add_vline(x=arl1, line_dash="dash", line_color=VERDE, annotation_text="ARL1 teórico",
                                 annotation_position="bottom right")
                fh.update_xaxes(title_text="Muestras hasta detectar el cambio")
                fh.update_yaxes(title_text="Frecuencia")
                st.plotly_chart(estilo(fh, "Distribución del largo de corrida tras el cambio", 340),
                                width="stretch", key="g_hist_rl")
                q = np.percentile(rl_arr, [10, 50, 90])
                st.write(f"ARL simulado = **{rl_arr.mean():.2f}** (± {rl_arr.std(ddof=1) / np.sqrt(reps):.2f}) | "
                         f"percentiles 10 / 50 / 90 = {q[0]:.0f} / {q[1]:.0f} / {q[2]:.0f} | "
                         f"truncadas: {trunc}")
                st.caption("La distribución es asimétrica: la mediana queda por debajo del ARL y a veces "
                           "tarda mucho más que el promedio.")

    # ---------------- 4. Curvas y tabla ----------------
    with t4:
        c = st.columns(2)
        modo = c[0].radio("Comparar curvas según", ["Tamaño de muestra n", "Amplitud L"], horizontal=True,
                          key="cv_m")
        eje = c[1].radio("Eje vertical", ["ARL1 (muestras)", "ATS1 (tiempo)"], horizontal=True, key="cv_e")
        grid = np.linspace(0.05, 3, 60)
        factor = h if eje.startswith("ATS") else 1.0
        fig = go.Figure()
        if modo.startswith("Tama"):
            ns = st.multiselect("Tamaños de muestra", OPCIONES_N,
                                default=[v for v in sorted({1, 5, 9, n}) if v in OPCIONES_N], key="cv_ns")
            for nn in ns:
                fig.add_trace(go.Scatter(x=grid, y=arl_shewhart(grid, nn, L) * factor, mode="lines",
                                         name=f"n = {nn}", line=dict(width=2.5)))
        else:
            Ls = st.multiselect("Valores de L", [2.0, 2.5, 2.75, 3.0, 3.25, 3.5], default=[2.5, 3.0, 3.5],
                                key="cv_Ls")
            for LL in Ls:
                fig.add_trace(go.Scatter(x=grid, y=arl_shewhart(grid, n, LL) * factor, mode="lines",
                                         name=f"L = {LL} (ARL0 ≈ {1 / (2 * stats.norm.sf(LL)):.0f})",
                                         line=dict(width=2.5)))
        fig.add_vline(x=delta, line_dash="dot", line_color=GRIS, annotation_text="δ configurado")
        fig.update_xaxes(title_text="Desplazamiento δ (en σ)")
        fig.update_yaxes(title_text=eje, type="log")
        st.plotly_chart(estilo(fig, f"{eje.split(' (')[0]} vs. desplazamiento", 420), width="stretch",
                        key="g_curvas")
        st.dataframe(tabla, hide_index=True)

    # ---------------- 5. n vs. frecuencia ----------------
    with t5:
        st.write("¿Conviene **muestras más grandes** o **muestrear más seguido**? Compare combinaciones "
                 "con una carga de inspección máxima.")
        c = st.columns(3)
        dd = c[0].number_input("δ a detectar (σ)", 0.1, 5.0, float(max(delta, 0.1)), 0.1, key="cm_d")
        Lc = c[1].number_input("L (Shewhart)", 1.0, 5.0, float(L), 0.05, key="cm_L")
        tope_c = c[2].number_input("Máx. unidades inspeccionadas por unidad de tiempo", 0.1, 10000.0, 5.0,
                                   0.5, key="cm_t")
        nlist = st.multiselect("Tamaños de muestra", OPCIONES_N, default=[1, 3, 5, 9], key="cm_n")
        hlist = st.multiselect("Intervalos de muestreo", [0.25, 0.5, 1.0, 2.0, 4.0, 8.0],
                               default=[0.5, 1.0, 2.0, 4.0], key="cm_h")
        arl0c = 1 / (2 * stats.norm.sf(Lc))
        filas = []
        for nn in nlist:
            for hh in hlist:
                a1 = float(arl_shewhart(dd, nn, Lc))
                filas.append({"n": nn, "h": hh, "Carga (n/h)": round(nn / hh, 2), "ARL1": round(a1, 2),
                              "ATS1": round(a1 * hh, 2), "ATS0": round(arl0c * hh, 1),
                              "Cumple carga": "Sí" if nn / hh <= tope_c else "No"})
        if filas:
            comp = pd.DataFrame(filas).sort_values("ATS1")
            st.dataframe(comp, hide_index=True)
            ok = comp[comp["Cumple carga"] == "Sí"]
            if len(ok):
                b = ok.iloc[0]
                st.success(f"Mejor opción dentro de la carga permitida: **n = {int(b['n'])}, h = {b['h']}** "
                           f"(ATS1 = {b['ATS1']}, ATS0 = {b['ATS0']}).")
            else:
                st.warning("Ninguna combinación cumple la carga máxima; aumente el tope o reduzca n.")
            st.caption("Reducir h baja el ATS1 pero también el ATS0 (más falsas alarmas por unidad de "
                       "tiempo). Aumentar n baja el ARL1 sin cambiar el ARL0.")
            st.session_state["arl_comp"] = comp