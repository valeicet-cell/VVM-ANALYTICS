"""Módulo 7. Muestreo de aceptación: planes simple, doble, múltiple y secuencial."""
import math

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats

from utils.graficos import AZUL, ROJO, VERDE, NARANJA, GRIS, estilo

DISTS = ["Binomial", "Poisson", "Hipergeométrica"]
PALETA = ["#0B6FD1", "#2BAE66", "#F59E0B", "#E5484D", "#6B84A3", "#0F2740", "#58AEFF", "#8A9BB0"]# (lote mínimo, lote máximo, letra código, tamaño de muestra): ANSI/ASQ Z1.4, nivel II general
Z14 = [(2, 8, "A", 2), (9, 15, "B", 3), (16, 25, "C", 5), (26, 50, "D", 8), (51, 90, "E", 13),
       (91, 150, "F", 20), (151, 280, "G", 32), (281, 500, "H", 50), (501, 1200, "J", 80),
       (1201, 3200, "K", 125), (3201, 10000, "L", 200), (10001, 35000, "M", 315),
       (35001, 150000, "N", 500), (150001, 500000, "P", 800), (500001, 10 ** 12, "Q", 1250)]


# ---------------------------------------------------------------- cálculos
def pa_simple(n, c, p, dist="Binomial", N=None):
    """Probabilidad de aceptación de un plan simple (n, c) para una fracción defectuosa p."""
    p = np.atleast_1d(np.asarray(p, float))
    if dist == "Binomial":
        return stats.binom.cdf(c, n, p)
    if dist == "Poisson":
        return stats.poisson.cdf(c, np.asarray(n) * p)
    D = np.round(N * p).astype(int)
    return stats.hypergeom.cdf(c, N, D, n)


def disenar_plan(aql, alfa, ltpd, beta, dist, N=None, n_max=5000, c_max=200):
    """Plan (n, c) de menor n con Pa(NAC) ≥ 1-α y Pa(NRC) ≤ β. Devuelve (n, c) o None."""
    ns = np.arange(1, n_max + 1)
    if dist == "Hipergeométrica":
        ns = ns[ns <= N]
    mejor = None
    for c in range(c_max + 1):
        ok = (pa_simple(ns, c, aql, dist, N) >= 1 - alfa) & (pa_simple(ns, c, ltpd, dist, N) <= beta)
        if ok.any():
            n = int(ns[ok][0])
            if mejor is None or n < mejor[0]:
                mejor = (n, c)
    return mejor


def pa_multiple(ns, ac, rc, p):
    """Plan por etapas (binomial). ns: tamaño de cada etapa; ac: números de aceptación acumulados;
    rc: números de rechazo acumulados. Devuelve (Pa, tamaño medio de muestra ASN).
    El plan doble es el caso de 2 etapas y el simple el de 1 etapa."""
    p = np.atleast_1d(np.asarray(p, float))
    pa = np.zeros_like(p)
    asn = np.zeros_like(p)
    cont = {0: np.ones_like(p)}  # defectuosos acumulados -> probabilidad de seguir en ese estado
    for ni, a, r in zip(ns, ac, rc):
        if not cont:
            break
        asn += sum(cont.values()) * ni
        nuevo = {}
        for d0, pr in cont.items():
            for k in range(0, min(ni, r - d0) + 1):
                prob = pr * stats.binom.pmf(k, ni, p)
                d = d0 + k
                if d <= a:
                    pa += prob
                elif d < r:
                    nuevo[d] = nuevo.get(d, 0.0) + prob
        cont = nuevo
    return pa, asn


def evaluar(plan, p, dist, N):
    """(Pa, ASN) de cualquier plan guardado como dict con ns, ac y rc."""
    p = np.atleast_1d(np.asarray(p, float))
    if len(plan["ns"]) == 1:
        return pa_simple(plan["ns"][0], plan["ac"][0], p, dist, N), np.full_like(p, float(plan["ns"][0]))
    return pa_multiple(plan["ns"], plan["ac"], plan["rc"], p)


def validar_plan(ns, ac, rc):
    """Texto de error si el plan por etapas es inconsistente; None si es válido."""
    if any(a < 0 for a in ac):
        return "Los números de aceptación no pueden ser negativos."
    if any(r <= a for a, r in zip(ac, rc)):
        return "En cada etapa el número de rechazo debe ser mayor que el de aceptación."
    if any(ac[i] > ac[i + 1] for i in range(len(ac) - 1)):
        return "Los números de aceptación acumulados no pueden disminuir de una etapa a otra."
    if any(rc[i] > rc[i + 1] for i in range(len(rc) - 1)):
        return "Los números de rechazo acumulados no pueden disminuir de una etapa a otra."
    if rc[-1] != ac[-1] + 1:
        return "En la última etapa el número de rechazo debe ser igual al de aceptación + 1."
    return None


def plan_simple(n, c):
    return {"tipo": "Simple", "nombre": f"Simple n={n}, c={c}", "ns": [n], "ac": [c], "rc": [c + 1]}


def plan_doble(n1, c1, r1, n2, c2):
    return {"tipo": "Doble", "nombre": f"Doble {n1}/{n2} (c1={c1}, r1={r1}, c2={c2})",
            "ns": [n1, n2], "ac": [c1, c2], "rc": [r1, c2 + 1]}


def plan_multiple(k, ni, ac, rc):
    return {"tipo": "Múltiple", "nombre": f"Múltiple {k} etapas × {ni}", "ns": [ni] * k,
            "ac": list(ac), "rc": list(rc)}


def params_secuencial(p1, p2, alfa, beta):
    """Prueba secuencial de razón de probabilidades (Wald): h1, h2 y pendiente s."""
    k = math.log(p2 * (1 - p1) / (p1 * (1 - p2)))
    return (math.log((1 - alfa) / beta) / k, math.log((1 - beta) / alfa) / k,
            math.log((1 - p1) / (1 - p2)) / k)


def arbol_doble(n1, c1, r1, n2, c2, p):
    """Árbol de probabilidades del plan doble para una fracción defectuosa p."""
    pa1 = float(stats.binom.cdf(c1, n1, p))
    pr1 = float(stats.binom.sf(r1 - 1, n1, p))
    inter = []
    for d1 in range(c1 + 1, r1):
        inter.append((d1, float(stats.binom.pmf(d1, n1, p)), float(stats.binom.cdf(c2 - d1, n2, p))))
    L = 2 + 2 * len(inter)
    y_a1, y_r1 = L - 1.0, 0.0
    ys_inter = [(L - 2 - 2 * j + L - 3 - 2 * j) / 2 for j in range(len(inter))]
    yr = (y_a1 + y_r1 + sum(ys_inter)) / (2 + len(inter))

    texto = "#0F2740"        # color de letra: azul marino oscuro, legible sobre los cuadros claros
    # tipo de nodo -> (color de relleno del cuadro, color del borde / punto)
    estilos = {"azul": ("#DCEBFA", "#0B6FD1"), "verde": ("#D6F2E3", "#1E9E58"),
               "rojo": ("#FBDADC", "#D93A40"), "naranja": ("#FDEBC8", "#D98A04")}

    lx, ly, etiquetas, nodos, filas = [], [], [], [], []

    def rama(x0, y0, x1, y1, txt):
        lx.extend([x0, x1, None])
        ly.extend([y0, y1, None])
        etiquetas.append(((x0 + x1) / 2, (y0 + y1) / 2, txt))

    nodos.append((0, yr, f"<b>1.ª muestra</b><br>n₁ = {n1}", "azul", "arriba"))
    rama(0, yr, 1, y_a1, f"d₁ ≤ {c1}<br>{pa1:.4f}")
    nodos.append((1, y_a1, f"<b>ACEPTA</b>  P = {pa1:.4f}", "verde", "derecha"))
    filas.append({"Camino": f"d₁ ≤ {c1}", "Probabilidad": pa1, "Decisión": "Acepta"})
    for j, (d1, pd1, pa2) in enumerate(inter):
        y_n = ys_inter[j]
        y_acc, y_rej = L - 2 - 2 * j, L - 3 - 2 * j
        rama(0, yr, 1, y_n, f"d₁ = {d1}<br>{pd1:.4f}")
        nodos.append((1, y_n, f"<b>2.ª muestra</b> (n₂ = {n2})", "naranja", "arriba"))
        cond = f"d₂ ≤ {c2 - d1}" if c2 - d1 >= 0 else "no puede aceptarse"
        rama(1, y_n, 2, y_acc, f"{cond}<br>{pa2:.4f}")
        rama(1, y_n, 2, y_rej, f"d₂ > {max(c2 - d1, 0) if c2 - d1 >= 0 else 0}<br>{1 - pa2:.4f}")
        nodos.append((2, y_acc, f"<b>ACEPTA</b>  {pd1:.4f} × {pa2:.4f} = {pd1 * pa2:.4f}", "verde", "derecha"))
        nodos.append((2, y_rej, f"<b>RECHAZA</b>  {pd1:.4f} × {1 - pa2:.4f} = {pd1 * (1 - pa2):.4f}", "rojo",
                      "derecha"))
        filas.append({"Camino": f"d₁ = {d1} y d₂ ≤ {c2 - d1}", "Probabilidad": pd1 * pa2, "Decisión": "Acepta"})
        filas.append({"Camino": f"d₁ = {d1} y d₂ > {c2 - d1}", "Probabilidad": pd1 * (1 - pa2),
                      "Decisión": "Rechaza"})
    rama(0, yr, 1, y_r1, f"d₁ ≥ {r1}<br>{pr1:.4f}")
    nodos.append((1, y_r1, f"<b>RECHAZA</b>  P = {pr1:.4f}", "rojo", "derecha"))
    filas.append({"Camino": f"d₁ ≥ {r1}", "Probabilidad": pr1, "Decisión": "Rechaza"})

    fig = go.Figure()
    # ramas
    fig.add_trace(go.Scatter(x=lx, y=ly, mode="lines", line=dict(color="#6B84A3", width=2),
                             hoverinfo="skip", showlegend=False))
    # puntos de cada nodo
    fig.add_trace(go.Scatter(x=[n[0] for n in nodos], y=[n[1] for n in nodos], mode="markers",
                             showlegend=False, hoverinfo="skip",
                             marker=dict(size=13, color=[estilos[n[3]][1] for n in nodos],
                                         line=dict(color="white", width=2))))
    # cuadro con texto para cada nodo (resultado de cada camino)
    for x, y, txt, tipo, lado in nodos:
        relleno, borde = estilos[tipo]
        if lado == "arriba":
            pos = dict(xanchor="center", yanchor="bottom", xshift=0, yshift=12)
        else:
            pos = dict(xanchor="left", yanchor="middle", xshift=14, yshift=0)
        fig.add_annotation(x=x, y=y, text=txt, showarrow=False, align="left",
                           font=dict(size=12, color=texto), bgcolor=relleno, bordercolor=borde,
                           borderwidth=1.5, borderpad=5, **pos)
    # cuadro con texto para cada rama (condición y probabilidad)
    for x, y, t in etiquetas:
        fig.add_annotation(x=x, y=y, text=t, showarrow=False, align="center",
                           font=dict(size=11, color=texto), bgcolor="white", bordercolor="#6B84A3",
                           borderwidth=1, borderpad=3)
    fig.update_xaxes(visible=False, range=[-0.4, 3.95])
    fig.update_yaxes(visible=False, range=[-0.8, L - 0.2])
    fig.update_layout(height=max(380, 75 * L + 120), margin=dict(l=5, r=5, t=15, b=5), showlegend=False,
                      paper_bgcolor="white", plot_bgcolor="white",
                      font=dict(family="Segoe UI, Arial, sans-serif", size=12, color=texto))
    pa_total = pa1 + sum(pd1 * pa2 for _, pd1, pa2 in inter)
    return fig, pd.DataFrame(filas), float(pa_total)


# ---------------------------------------------------------------- estado y callbacks
def leer_par():
    """Parámetros de la barra superior (también usable desde callbacks)."""
    ss = st.session_state
    return {"aql": ss.get("m7_aql", 1.0) / 100, "alfa": ss.get("m7_a", 0.05),
            "ltpd": ss.get("m7_ltpd", 6.0) / 100, "beta": ss.get("m7_b", 0.10),
            "N": int(ss.get("m7_N", 1000)), "dist": ss.get("m7_dist", "Binomial")}


def _disenar_cb():
    par = leer_par()
    if par["aql"] >= par["ltpd"]:
        st.session_state["m7_msg"] = ("error", "El NAC debe ser menor que el NRC/LTPD.")
        return
    plan = disenar_plan(par["aql"], par["alfa"], par["ltpd"], par["beta"], par["dist"], par["N"])
    if plan is None:
        st.session_state["m7_msg"] = ("error", "No se encontró un plan con n ≤ 5000. Relaje α o β, "
                                               "o separe más el NAC y el NRC.")
    else:
        st.session_state["m7_sn"], st.session_state["m7_sc"] = plan
        st.session_state["m7_msg"] = ("success", f"Plan de menor tamaño: n = {plan[0]}, c = {plan[1]}.")


def agregar_plan(plan):
    lista = st.session_state.setdefault("m7_planes", [])
    if any(q["nombre"] == plan["nombre"] for q in lista):
        st.toast("Ese plan ya está en la comparación.")
        return
    lista.append(plan)
    st.toast("Plan agregado a la comparación de curvas OC.")


def _quitar(nombre):
    st.session_state["m7_planes"] = [q for q in st.session_state.get("m7_planes", [])
                                     if q["nombre"] != nombre]


def _vaciar():
    st.session_state["m7_planes"] = []


def _nueva_semilla():
    st.session_state["m7_qs"] = int(st.session_state.get("m7_qs", 7)) + 1


# ---------------------------------------------------------------- gráficos y panel
def _fig(x, y, titulo, xt, yt, marcas=()):
    f = go.Figure(go.Scatter(x=x, y=y, mode="lines", line=dict(color=AZUL, width=2.5)))
    for mx, my, nom, color in marcas:
        f.add_trace(go.Scatter(x=[mx], y=[my], mode="markers+text", text=[nom], textposition="top right",
                               marker=dict(size=11, color=color, line=dict(color="white", width=1)),
                               showlegend=False))
    f.update_layout(showlegend=False)
    f.update_xaxes(title_text=xt)
    f.update_yaxes(title_text=yt)
    return estilo(f, titulo, 330)


def panel(plan, par, dist, clave, guardar=False):
    """Indicadores, curvas OC, AOQ, ATI (y ASN) de un plan simple, doble o múltiple."""
    ns, N = plan["ns"], par["N"]
    aql, ltpd, alfa, beta = par["aql"], par["ltpd"], par["alfa"], par["beta"]
    multi = len(ns) > 1
    if sum(ns) > N:
        st.error("El tamaño máximo de muestra supera el tamaño del lote.")
        return
    if multi and dist != "Binomial":
        st.caption("Los planes doble y múltiple se evalúan con la distribución binomial.")

    p = np.linspace(0, min(1.0, max(4 * ltpd, 0.15)), 300)
    pa, asn = evaluar(plan, p, dist, N)
    aoq = p * pa * (N - asn) / N
    ati = asn + (1 - pa) * (N - asn)
    pa_a, asn_a = (float(v[0]) for v in evaluar(plan, aql, dist, N))
    pa_l, asn_l = (float(v[0]) for v in evaluar(plan, ltpd, dist, N))

    m = st.columns(5)
    m[0].metric(f"α real (meta ≤ {alfa:g})", f"{1 - pa_a:.4f}", help="Riesgo tipo I: 1 − Pa en el NAC.")
    m[1].metric(f"β real (meta ≤ {beta:g})", f"{pa_l:.4f}", help="Riesgo tipo II: Pa en el NRC/LTPD.")
    m[2].metric("Pa en el NAC", f"{pa_a:.4f}")
    m[3].metric("Pa en el NRC/LTPD", f"{pa_l:.4f}")
    m[4].metric("AOQL", f"{aoq.max() * 100:.2f} %", help=f"Máximo de la AOQ, en p = {p[aoq.argmax()] * 100:.2f} %")
    k = st.columns(4)
    k[0].metric("ASN en el NAC", f"{asn_a:.1f}", help="Tamaño medio de muestra cuando la calidad es la NAC.")
    k[1].metric("ASN en el NRC/LTPD", f"{asn_l:.1f}")
    k[2].metric("Muestra máxima", f"{sum(ns)}")
    k[3].metric("Plan", plan["tipo"])
    if (1 - pa_a) <= alfa + 1e-9 and pa_l <= beta + 1e-9:
        st.success("El plan cumple los dos riesgos: α real ≤ α y β real ≤ β.")
    else:
        st.warning("El plan NO cumple al menos uno de los riesgos objetivo (α o β). Revise n y los "
                   "números de aceptación, o use el diseño automático del plan simple.")

    marcas = [(aql * 100, pa_a, "NAC", VERDE), (ltpd * 100, pa_l, "NRC/LTPD", ROJO)]
    cols = st.columns(2)
    cols[0].plotly_chart(_fig(p * 100, pa, "Curva OC", "% defectuoso (p)", "Prob. de aceptación", marcas),
                         width="stretch", key=f"oc_{clave}")
    cols[1].plotly_chart(_fig(p * 100, aoq * 100, "Calidad media de salida (AOQ)", "% defectuoso (p)",
                              "AOQ (%)"), width="stretch", key=f"aoq_{clave}")
    cols = st.columns(2)
    cols[0].plotly_chart(_fig(p * 100, ati, "Inspección total media (ATI)", "% defectuoso (p)",
                              "Unidades inspeccionadas"), width="stretch", key=f"ati_{clave}")
    if multi:
        cols[1].plotly_chart(_fig(p * 100, asn, "Tamaño medio de muestra (ASN)", "% defectuoso (p)", "ASN"),
                             width="stretch", key=f"asn_{clave}")
    pp = np.array(sorted({aql / 2, aql, 2 * aql, ltpd / 2, ltpd, 2 * ltpd}))
    pap, asnp = evaluar(plan, pp, dist, N)
    st.dataframe(pd.DataFrame({"p (%)": (pp * 100).round(3), "Pa": pap.round(4), "ASN": asnp.round(1),
                               "AOQ (%)": (pp * pap * (N - asnp) / N * 100).round(4),
                               "ATI": (asnp + (1 - pap) * (N - asnp)).round(1)}), hide_index=True)
    st.markdown("**Análisis del comportamiento del plan**")
    st.info(f"Para el nivel NAC de {aql*100:.2f} %, la probabilidad de aceptación calculada es {pa_a:.4f}; "
            f"para el nivel NRC/LTPD de {ltpd*100:.2f} %, es {pa_l:.4f}. El tamaño medio de muestra varía "
            f"entre {asn_a:.1f} y {asn_l:.1f} unidades en esos escenarios y el AOQL estimado es "
            f"{aoq.max()*100:.2f} %. Estos indicadores permiten valorar conjuntamente la protección "
            "al productor y al consumidor, así como el esfuerzo de inspección requerido.")
    st.button("➕ Agregar a la comparación de curvas OC", on_click=agregar_plan, args=(plan,),
              key=f"add_{clave}")
    if guardar and not multi:
        st.session_state["plan_aceptacion"] = {
            "n": ns[0], "c": plan["ac"][0], "N": N, "dist": dist, "aql": aql, "ltpd": ltpd,
            "alfa": alfa, "beta": beta, "pa_aql": pa_a, "pa_ltpd": pa_l, "aoql": float(aoq.max())}


# ---------------------------------------------------------------- pantalla
def mostrar():
    st.header("Módulo 7. Muestreo de aceptación")

    # ---- Barra de parámetros ----
    with st.container(border=True):
        c = st.columns(6)
        c[0].number_input("NAC / AQL (%)", 0.01, 50.0, 1.0, 0.1, key="m7_aql",
                          help="Nivel aceptable de calidad: % defectuoso que se considera bueno.")
        c[1].number_input("α · riesgo tipo I", 0.001, 0.5, 0.05, 0.01, format="%.3f", key="m7_a",
                          help="Riesgo del productor: prob. de rechazar un lote de calidad NAC.")
        c[2].number_input("NRC / LTPD (%)", 0.1, 90.0, 6.0, 0.5, key="m7_ltpd",
                          help="Nivel rechazable de calidad: % defectuoso que se quiere rechazar casi siempre.")
        c[3].number_input("β · riesgo tipo II", 0.001, 0.5, 0.10, 0.01, format="%.3f", key="m7_b",
                          help="Riesgo del consumidor: prob. de aceptar un lote de calidad NRC/LTPD.")
        c[4].number_input("Tamaño del lote N", 10, 10_000_000, 1000, 100, key="m7_N")
        c[5].selectbox("Distribución (plan simple)", DISTS, key="m7_dist")
    par = leer_par()
    aql, alfa, ltpd, beta, N, dist = (par[k] for k in ("aql", "alfa", "ltpd", "beta", "N", "dist"))
    if aql >= ltpd:
        st.error("El NAC debe ser menor que el NRC/LTPD.")
        return
    cards = st.columns(5)
    cards[0].metric("α · riesgo del productor", f"{alfa:g}", help="Riesgo tipo I")
    cards[1].metric("β · riesgo del consumidor", f"{beta:g}", help="Riesgo tipo II")
    cards[2].metric("NAC (AQL)", f"{aql * 100:g} %", help="Aquí la curva OC debe valer 1 − α.")
    cards[3].metric("NRC (RQL)", f"{ltpd * 100:g} %", help="Aquí la curva OC debe valer β.")
    cards[4].metric("LTPD", f"{ltpd * 100:g} %", help="Mismo punto que el NRC en este módulo.")

    t_s, t_d, t_m, t_q, t_c, t_n = st.tabs([
        "Plan simple", "Plan doble", "Plan múltiple", "Plan secuencial",
        "Comparar curvas OC", "Norma MIL-STD / Z1.4"])

    # ---------------- Simple ----------------
    with t_s:
        st.markdown("**Plan simple**")
        st.caption("Se inspecciona una única muestra de tamaño n. El lote se acepta si el número de unidades defectuosas es menor o igual que c; de lo contrario, se rechaza. Es un procedimiento directo y fácil de administrar.")
        st.session_state.setdefault("m7_sn", 80)
        st.session_state.setdefault("m7_sc", 2)
        c = st.columns([1, 1.3, 2])
        n = int(c[0].number_input("n (tamaño de muestra)", 1, 100000, step=1, key="m7_sn"))
        cc = int(c[1].number_input("c (número de aceptación)", 0, 1000, step=1, key="m7_sc"))
        c[2].write("")
        c[2].button("🧮 Diseñar plan con los parámetros de arriba", on_click=_disenar_cb, key="m7_dis",
                    help="Busca el menor n con Pa(NAC) ≥ 1 − α y Pa(NRC) ≤ β.")
        msg = st.session_state.pop("m7_msg", None)
        if msg:
            getattr(st, msg[0])(msg[1])
        panel(plan_simple(n, cc), par, dist, "s", guardar=True)

    # ---------------- Doble ----------------
    with t_d:
        st.markdown("**Plan doble**")
        st.caption("La decisión se toma en hasta dos etapas: la primera muestra puede permitir aceptar, rechazar o continuar con una segunda muestra. La decisión final considera el número acumulado de defectuosos.")
        for k_, v_ in {"m7_n1": 50, "m7_c1": 1, "m7_r1": 4, "m7_n2": 100, "m7_c2": 3}.items():
            st.session_state.setdefault(k_, v_)
        st.caption("Se toma n₁: si d₁ ≤ c₁ se acepta; si d₁ ≥ r₁ se rechaza; si no, se toma n₂ y se acepta cuando "
                   "d₁ + d₂ ≤ c₂. Los valores iniciales son un ejemplo: ingrese los de su plan.")
        c = st.columns(5)
        n1 = int(c[0].number_input("n₁", 1, 10000, step=1, key="m7_n1"))
        c1 = int(c[1].number_input("c₁", 0, 100, step=1, key="m7_c1"))
        r1 = int(c[2].number_input("r₁", 1, 101, step=1, key="m7_r1"))
        n2 = int(c[3].number_input("n₂", 1, 10000, step=1, key="m7_n2"))
        c2 = int(c[4].number_input("c₂", 0, 200, step=1, key="m7_c2"))
        plan = plan_doble(n1, c1, r1, n2, c2)
        err = validar_plan(plan["ns"], plan["ac"], plan["rc"])
        if err:
            st.error(err + " (Debe cumplirse c₁ < r₁ ≤ c₂ + 1.)")
        else:
            panel(plan, par, dist, "d")
            st.subheader("Árbol de probabilidades")
            op = st.radio("Calidad del lote para calcular el árbol", ["NAC", "NRC/LTPD", "Otro valor"],
                          horizontal=True, key="m7_tp")
            if op == "NAC":
                pt = aql
            elif op == "NRC/LTPD":
                pt = ltpd
            else:
                pt = st.number_input("p (% defectuoso)", 0.01, 90.0, 3.0, 0.5, key="m7_tpo") / 100
            fig_a, tabla_a, pa_t = arbol_doble(n1, c1, r1, n2, c2, pt)
            st.plotly_chart(fig_a, width="stretch", key="g_arbol", theme=None)
            k = st.columns(3)
            k[0].metric("p usado", f"{pt * 100:.3f} %")
            k[1].metric("Prob. de aceptar el lote", f"{pa_t:.4f}")
            k[2].metric("Prob. de rechazarlo", f"{1 - pa_t:.4f}")
            st.caption("La probabilidad de aceptación es la suma de las hojas verdes. En las ramas de segunda "
                       "muestra, cada hoja es el producto P(d₁) × P(d₂ | d₁). Cálculo binomial.")
            tabla_a["Probabilidad"] = tabla_a["Probabilidad"].round(5)
            st.dataframe(tabla_a, hide_index=True)

    # ---------------- Múltiple ----------------
    with t_m:
        st.markdown("**Plan múltiple**")
        st.caption("El lote se evalúa mediante varias muestras sucesivas de tamaño reducido. En cada etapa, el acumulado de defectuosos determina si se acepta, se rechaza o se continúa inspeccionando.")
        st.caption("Plan por etapas con igual tamaño de muestra en cada una. En cada etapa se compara el total "
                   "ACUMULADO de defectuosos con el número de aceptación (Ac) y de rechazo (Re) de esa etapa. "
                   "Los valores iniciales son un ejemplo, no provienen de ninguna norma.")
        c = st.columns(3)
        k = int(c[0].number_input("Número de etapas", 2, 7, 5, 1, key="m7_mk"))
        ni = int(c[1].number_input("Tamaño de muestra por etapa", 1, 10000, 20, 1, key="m7_mn"))
        base = pd.DataFrame({"Etapa": list(range(1, k + 1)),
                             "Ac (acumulado)": list(range(k)),
                             "Re (acumulado)": [min(i + 3, k) for i in range(k)]})
        ed = st.data_editor(base, hide_index=True, disabled=["Etapa"], key=f"m7_mt_{k}")
        ed = ed.apply(pd.to_numeric, errors="coerce")
        if ed.isna().any().any():
            st.error("Complete todos los números de aceptación y rechazo.")
        else:
            ac = ed["Ac (acumulado)"].astype(int).tolist()
            rc = ed["Re (acumulado)"].astype(int).tolist()
            err = validar_plan([ni] * k, ac, rc)
            if err:
                st.error(err)
            else:
                st.caption("n acumulado por etapa: " + ", ".join(str(ni * (i + 1)) for i in range(k)))
                panel(plan_multiple(k, ni, ac, rc), par, dist, "m")

    # ---------------- Secuencial ----------------
    with t_q:
        st.markdown("**Plan secuencial**")
        st.caption("Las unidades se inspeccionan sucesivamente y la decisión se actualiza con la evidencia acumulada: aceptar, rechazar o continuar. Puede reducir la inspección promedio, aunque exige una administración más rigurosa.")
        h1, h2, s = params_secuencial(aql, ltpd, alfa, beta)
        st.caption("Prueba secuencial de razón de probabilidades (Wald), con p₁ = NAC y p₂ = NRC/LTPD. Tras inspeccionar "
                   "n unidades: se **acepta** si los defectuosos acumulados d ≤ −h₁ + s·n; se **rechaza** si "
                   "d ≥ h₂ + s·n; en caso contrario se sigue inspeccionando.")
        m = st.columns(4)
        m[0].metric("h₁", f"{h1:.4f}")
        m[1].metric("h₂", f"{h2:.4f}")
        m[2].metric("Pendiente s", f"{s:.5f}")
        m[3].metric("Mín. unidades para aceptar", f"{math.ceil(h1 / s)}",
                    help="Menor n con el que se puede aceptar sin ningún defectuoso.")
        st.session_state.setdefault("m7_qp", 1.0)
        st.session_state.setdefault("m7_qs", 7)
        c = st.columns(4)
        nmax = int(c[0].number_input("Máx. unidades a inspeccionar", 20, 5000, 150, 10, key="m7_qmax"))
        p_real = c[1].number_input("% defectuoso real del lote simulado", 0.0, 100.0, step=0.5, key="m7_qp") / 100
        c[2].number_input("Semilla", 0, 999999, step=1, key="m7_qs")
        c[3].write("")
        c[3].button("🎲 Nueva inspección", on_click=_nueva_semilla)

        rng = np.random.default_rng(int(st.session_state["m7_qs"]))
        d, xs, ys, dec = 0, [0], [0], None
        for i in range(1, nmax + 1):
            d += int(rng.random() < p_real)
            xs.append(i)
            ys.append(d)
            if d <= -h1 + s * i:
                dec = ("ACEPTAR", i)
                break
            if d >= h2 + s * i:
                dec = ("RECHAZAR", i)
                break
        nn = np.arange(0, nmax + 1)
        A, R = -h1 + s * nn, h2 + s * nn
        ymax = float(R[-1]) * 1.05
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=nn, y=A, mode="lines", name="Línea de aceptación",
                                 line=dict(color=VERDE, width=2.5), fill="tozeroy",
                                 fillcolor="rgba(43,174,102,.12)"))
        fig.add_trace(go.Scatter(x=nn, y=np.full_like(R, ymax), mode="lines", line=dict(width=0),
                                 hoverinfo="skip", showlegend=False))
        fig.add_trace(go.Scatter(x=nn, y=R, mode="lines", name="Línea de rechazo",
                                 line=dict(color=ROJO, width=2.5), fill="tonexty",
                                 fillcolor="rgba(229,72,77,.10)"))
        fig.add_trace(go.Scatter(x=xs, y=ys, mode="lines", name="Defectuosos acumulados",
                                 line=dict(color=AZUL, width=2, shape="hv")))
        if dec:
            fig.add_trace(go.Scatter(x=[dec[1]], y=[ys[-1]], mode="markers+text", showlegend=False,
                                     text=[dec[0]], textposition="top center",
                                     marker=dict(size=13, color=VERDE if dec[0] == "ACEPTAR" else ROJO,
                                                 line=dict(color="white", width=2))))
        fig.add_annotation(x=nmax * 0.12, y=ymax * 0.82, text="Rechazar", showarrow=False,
                           font=dict(color=ROJO, size=13))
        fig.add_annotation(x=nmax * 0.82, y=ymax * 0.10, text="Aceptar", showarrow=False,
                           font=dict(color=VERDE, size=13))
        fig.add_annotation(x=nmax * 0.5, y=(h2 + s * nmax * 0.5 + max(-h1 + s * nmax * 0.5, 0)) / 2,
                           text="Continuar inspeccionando", showarrow=False, font=dict(color="#4A4F75"))
        fig.update_xaxes(title_text="Unidades inspeccionadas (n)", range=[0, nmax])
        fig.update_yaxes(title_text="Defectuosos acumulados (d)", range=[0, ymax])
        st.plotly_chart(estilo(fig, "Plan secuencial: regiones de decisión", 440), width="stretch",
                        key="g_seq")
        if dec:
            st.success(f"Con un lote de {p_real * 100:g} % defectuoso, la decisión fue **{dec[0]}** tras "
                       f"inspeccionar {dec[1]} unidades ({ys[-1]} defectuosas).")
        else:
            st.info(f"Sin decisión tras {nmax} unidades. En la práctica se fija un tamaño de truncamiento.")
        st.caption("Una sola inspección es aleatoria: pulse «Nueva inspección» varias veces. Con calidad NAC casi "
                   "siempre acepta rápido y con calidad NRC casi siempre rechaza. Los límites son los de Wald "
                   "(aproximados); los planes secuenciales normalizados incluyen un truncamiento.")

    # ---------------- Comparar curvas OC ----------------
    with t_c:
        st.caption("Agregue planes con el botón «Agregar a la comparación» de las otras pestañas, o escriba un "
                   "plan simple aquí. Todas las curvas se dibujan en la misma gráfica.")
        c = st.columns([1, 1, 1.4, 3])
        qn = int(c[0].number_input("n", 1, 100000, 100, 1, key="m7_qn"))
        qc = int(c[1].number_input("c", 0, 1000, 2, 1, key="m7_qc"))
        c[2].write("")
        c[2].button("Agregar plan simple", on_click=agregar_plan, args=(plan_simple(qn, qc),), key="m7_qadd")
        planes = st.session_state.get("m7_planes", [])
        if not planes:
            st.info("Aún no hay planes en la comparación.")
        else:
            nombres = [q["nombre"] for q in planes]
            sel = st.multiselect("Planes a mostrar", nombres, default=nombres, key="m7_sel")
            pmax = st.slider("p máximo del gráfico (%)", 1.0, 50.0,
                             float(min(50.0, max(10.0, round(3 * ltpd * 100)))), 1.0, key="m7_pmax") / 100
            p = np.linspace(0, pmax, 300)
            fig = go.Figure()
            filas = []
            for i, q in enumerate(planes):
                if q["nombre"] not in sel:
                    continue
                if sum(q["ns"]) > N:
                    st.warning(f"El plan «{q['nombre']}» tiene una muestra mayor que el lote N; se omite.")
                    continue
                pa, _ = evaluar(q, p, dist, N)
                fig.add_trace(go.Scatter(x=p * 100, y=pa, mode="lines", name=q["nombre"],
                                         line=dict(color=PALETA[i % len(PALETA)], width=2.5)))
                pa_a, asn_a = (float(v[0]) for v in evaluar(q, aql, dist, N))
                pa_l, asn_l = (float(v[0]) for v in evaluar(q, ltpd, dist, N))
                filas.append({"Plan": q["nombre"], "Tipo": q["tipo"], "Muestra máx.": sum(q["ns"]),
                              "Pa(NAC)": round(pa_a, 4), "α real": round(1 - pa_a, 4),
                              "Pa(NRC)": round(pa_l, 4), "β real": round(pa_l, 4),
                              "ASN(NAC)": round(asn_a, 1), "ASN(NRC)": round(asn_l, 1),
                              "¿Cumple α y β?": "Sí" if (1 - pa_a) <= alfa + 1e-9 and pa_l <= beta + 1e-9 else "No"})
            fig.add_vline(x=aql * 100, line_dash="dot", line_color=VERDE, annotation_text="NAC")
            fig.add_vline(x=ltpd * 100, line_dash="dot", line_color=ROJO, annotation_text="NRC/LTPD")
            fig.add_hline(y=1 - alfa, line_dash="dot", line_color=GRIS, annotation_text=f"1 − α = {1 - alfa:g}")
            fig.add_hline(y=beta, line_dash="dot", line_color=GRIS, annotation_text=f"β = {beta:g}")
            fig.add_trace(go.Scatter(x=[aql * 100], y=[1 - alfa], mode="markers", name="Punto del productor (NAC, 1−α)",
                                     marker=dict(size=13, color=VERDE, symbol="diamond", line=dict(color="white", width=2))))
            fig.add_trace(go.Scatter(x=[ltpd * 100], y=[beta], mode="markers", name="Punto del consumidor (NRC, β)",
                                     marker=dict(size=13, color=ROJO, symbol="diamond", line=dict(color="white", width=2))))
            fig.update_xaxes(title_text="% defectuoso (p)")
            fig.update_yaxes(title_text="Probabilidad de aceptación (Pa)", range=[0, 1.02])
            st.plotly_chart(estilo(fig, "Curvas OC comparadas", 500), width="stretch", key="g_oc_comp")
            st.caption("Un buen plan pasa por encima del punto verde y por debajo del rojo. Dos planes con curvas "
                       "casi iguales dan la misma protección; elija el que tenga menor ASN y sea más fácil de aplicar.")
            if filas:
                st.dataframe(pd.DataFrame(filas), hide_index=True)
            c = st.columns([2, 1, 1])
            quitar = c[0].selectbox("Quitar un plan de la comparación", nombres, key="m7_quitar")
            c[1].write("")
            c[1].button("Quitar", on_click=_quitar, args=(quitar,), key="m7_qb")
            c[2].write("")
            c[2].button("Vaciar comparación", on_click=_vaciar, key="m7_vac")

    # ---------------- Norma ----------------
    with t_n:
        st.caption("Inspección normal, nivel general II (ANSI/ASQ Z1.4). El número de aceptación (Ac) "
                   "se lee de la tabla II-A de la norma, cruzando la letra código con el NAC.")
        c = st.columns(5)
        Nl = int(c[0].number_input("Tamaño del lote", 2, 10_000_000, 1000, 100, key="m7_zN"))
        letra, nz = next((lt, sz) for lo, hi, lt, sz in Z14 if lo <= Nl <= hi)
        c[1].metric("Letra código", letra)
        c[2].metric("Tamaño de muestra", nz)
        ac = int(c[3].number_input("Ac (de la tabla de la norma)", 0, 100, 1, 1, key="m7_zac"))
        c[4].metric("Re", ac + 1)
        plan_z = plan_simple(min(nz, Nl), ac)
        plan_z["nombre"] = f"Z1.4 letra {letra}: n={min(nz, Nl)}, Ac={ac}"
        plan_z["tipo"] = "Simple (norma)"
        panel(plan_z, dict(par, N=Nl), "Binomial", "z")