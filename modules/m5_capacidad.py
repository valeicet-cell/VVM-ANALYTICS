"""Módulo 5. Capacidad del proceso: Cp, Cpk, Cpl, Cpu, Pp, Ppk y fracción no conforme."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from scipy import stats
from utils.graficos import estilo

from modules import m2_variables as m2
from utils.datos import datos_estables, tipo_grafico_por_defecto


# ---------------------------------------------------------------- cálculos
def sigma_dentro(df, col):
    """σ dentro de subgrupos, estimada con R̄/d2, S̄/c4 o MR̄/d2 según el tamaño."""
    tipo = tipo_grafico_por_defecto(df, col)
    principal, disp, n = m2.construir_series(df, col, tipo)
    lim = m2.calcular_limites(tipo, principal, disp, n)
    return lim["sigma"], tipo, n


def indices(mu, sigma, lei, les, pref):
    """Índices Cp/Cpu/Cpl/Cpk (pref='C', σ dentro) o Pp/Ppu/Ppl/Ppk (pref='P', σ global)."""
    r = {}
    if lei is not None and les is not None:
        r[f"{pref}p"] = (les - lei) / (6 * sigma)
    if les is not None:
        r[f"{pref}pu"] = (les - mu) / (3 * sigma)
    if lei is not None:
        r[f"{pref}pl"] = (mu - lei) / (3 * sigma)
    r[f"{pref}pk"] = min(r[k] for k in (f"{pref}pu", f"{pref}pl") if k in r)
    return r


def fnc(mu, sigma, lei, les):
    """Fracción no conforme esperada (debajo del LEI, encima del LES) bajo normalidad."""
    abajo = stats.norm.cdf((lei - mu) / sigma) if lei is not None else 0.0
    arriba = stats.norm.sf((les - mu) / sigma) if les is not None else 0.0
    return float(abajo), float(arriba)


def ic_cp(cp, n, conf=0.95):
    """Intervalo de confianza para Cp/Pp (chi-cuadrado)."""
    nu, a = n - 1, 1 - conf
    return (cp * np.sqrt(stats.chi2.ppf(a / 2, nu) / nu),
            cp * np.sqrt(stats.chi2.ppf(1 - a / 2, nu) / nu))


def ic_cpk(cpk, n, conf=0.95):
    """Intervalo aproximado para Cpk/Ppk (Bissell)."""
    z = stats.norm.ppf(1 - (1 - conf) / 2)
    e = z * np.sqrt(1 / (9 * n) + cpk ** 2 / (2 * (n - 1)))
    return cpk - e, cpk + e


def interpretar(cpk):
    """Devuelve (nivel de alerta, texto) según el valor de Cpk."""
    if cpk < 1.0:
        return "error", f"Cpk = {cpk:.2f} < 1,00: el proceso NO es capaz de cumplir las especificaciones."
    if cpk < 1.33:
        return "warning", f"Cpk = {cpk:.2f} (1,00 a 1,33): proceso marginalmente capaz; requiere control estricto."
    if cpk < 1.67:
        return "success", f"Cpk = {cpk:.2f} (1,33 a 1,67): proceso capaz."
    return "success", f"Cpk = {cpk:.2f} ≥ 1,67: proceso muy capaz."


def _fmt(v):
    return "—" if v is None else f"{v:.3f}"


# ---------------------------------------------------------------- pantalla
def mostrar():
    st.header("Módulo 5. Capacidad del proceso")
    df = st.session_state.get("df")
    col = st.session_state.get("col_valor")
    if df is None:
        st.warning("Primero cargue datos en **Gestión de datos** (Módulo 1).")
        return

    usar = True
    lim_m2 = st.session_state.get("limites")
    if lim_m2 and lim_m2.get("excluidos"):
        usar = st.checkbox("Excluir los puntos retirados en la Fase I (Módulo 2)", value=True)
    st.caption("La capacidad solo tiene sentido si el proceso es estable (revise los Módulos 2 y 4).")
    d = datos_estables(df, usar).dropna(subset=[col]).copy()
    x0 = d[col]

    # ---- Configuración (barra superior compacta) ----
    with st.container(border=True):
        c = st.columns([1.2, 1.2, 1])
        tiene_lei = c[0].checkbox("Tiene límite inferior (LEI)", True)
        lei = c[0].number_input("LEI", value=float(round(x0.mean() - 4 * x0.std(), 4)),
                                step=0.1, format="%.4f", disabled=not tiene_lei, key="cap_lei")
        tiene_les = c[1].checkbox("Tiene límite superior (LES)", True)
        les = c[1].number_input("LES", value=float(round(x0.mean() + 4 * x0.std(), 4)),
                                step=0.1, format="%.4f", disabled=not tiene_les, key="cap_les")
        transf = c[2].checkbox("Transformar con Box-Cox",
                               help="Úselo si los datos no son normales (requiere valores positivos).")
    st.caption("Los valores por defecto de LEI y LES son solo de referencia: ingrese las "
               "especificaciones reales.")
    lei = lei if tiene_lei else None
    les = les if tiene_les else None
    if lei is None and les is None:
        st.error("Defina al menos un límite de especificación.")
        return
    if lei is not None and les is not None and lei >= les:
        st.error("El LEI debe ser menor que el LES.")
        return

    # ---- Transformación opcional ----
    lam = None
    if transf:
        if (d[col] <= 0).any() or any(v is not None and v <= 0 for v in (lei, les)):
            st.error("Box-Cox requiere datos y especificaciones positivos.")
            return
        lam = float(stats.boxcox(d[col])[1])
        f = np.log if abs(lam) < 1e-9 else (lambda v: (np.power(v, lam) - 1) / lam)
        d[col] = f(d[col])
        lei = float(f(lei)) if lei is not None else None
        les = float(f(les)) if les is not None else None
        st.caption(f"λ de Box-Cox = {lam:.4f}. Los índices se calculan en la escala transformada.")
    x = d[col]
    mu, so, nobs = float(x.mean()), float(x.std(ddof=1)), len(x)
    try:
        sw, tipo_sw, n_sg = sigma_dentro(d, col)
    except Exception:
        sw, tipo_sw, n_sg = so, "global", 1
        st.warning("No se pudo estimar σ dentro de subgrupos; se usa σ global.")

    idx_c = indices(mu, sw, lei, les, "C")
    idx_p = indices(mu, so, lei, les, "P")
    st.caption(f"n = {nobs} | media = {mu:.4f} | σ dentro ({tipo_sw}) = {sw:.4f} | σ global = {so:.4f}")

    # ---- Índices ----
    st.subheader("2. Índices de capacidad")
    m = st.columns(4)
    for cc, (nom, val) in zip(m, (("Cp", idx_c.get("Cp")), ("Cpk", idx_c["Cpk"]),
                                  ("Pp", idx_p.get("Pp")), ("Ppk", idx_p["Ppk"]))):
        cc.metric(nom, _fmt(val))
    filas = []
    for r, sigma_nom in ((idx_c, "σ dentro"), (idx_p, "σ global")):
        for k, v in r.items():
            lo = hi = None
            if k in ("Cp", "Pp"):
                lo, hi = ic_cp(v, nobs)
            elif k in ("Cpk", "Ppk"):
                lo, hi = ic_cpk(v, nobs)
            filas.append({"Índice": k, "Valor": round(v, 3), "Sigma usada": sigma_nom,
                          "IC 95 % inferior": None if lo is None else round(lo, 3),
                          "IC 95 % superior": None if hi is None else round(hi, 3)})
    tabla_indices = pd.DataFrame(filas)
    st.dataframe(tabla_indices, hide_index=True)
    st.caption("Cp/Cpk (potencial) usan σ dentro de subgrupos; Pp/Ppk (desempeño real) usan σ global. "
               "Los IC son aproximados.")

    # ---- Fracción no conforme ----
    st.subheader("3. Fracción no conforme")
    bw, aw = fnc(mu, sw, lei, les)
    bg, ag = fnc(mu, so, lei, les)
    ob = float((x < lei).mean()) if lei is not None else 0.0
    oa = float((x > les).mean()) if les is not None else 0.0
    tabla_fnc = pd.DataFrame({
        "Concepto (PPM)": ["Por debajo del LEI", "Por encima del LES", "Total"],
        "Esperada, σ dentro": [bw * 1e6, aw * 1e6, (bw + aw) * 1e6],
        "Esperada, σ global": [bg * 1e6, ag * 1e6, (bg + ag) * 1e6],
        "Observada en los datos": [ob * 1e6, oa * 1e6, (ob + oa) * 1e6]}).round(1)
    st.dataframe(tabla_fnc, hide_index=True)
    st.write(f"Fracción no conforme esperada con σ global: **{(bg + ag) * 100:.4f} %** "
             f"({(bg + ag) * 1e6:.1f} PPM).")

    # ---- Histograma ----
    xs = np.linspace(x.min() - 3 * so, x.max() + 3 * so, 300)
    fig = go.Figure(go.Histogram(x=x, histnorm="probability density", marker_color="#9CC8F2",
                                 name="Datos"))
    fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, mu, so), mode="lines",
                             line=dict(color="#0B6FD1", width=2.5), name="Normal (σ global)"))
    for v, nom in ((lei, "LEI"), (les, "LES")):
        if v is not None:
            fig.add_vline(x=v, line_color="#E5484D", line_dash="dash", annotation_text=nom)
    fig.add_vline(x=mu, line_color="#2BAE66", annotation_text="Media")
    fig = estilo(fig, "Distribución del proceso frente a las especificaciones", 420)
    st.plotly_chart(fig, width="stretch")

    # ---- Interpretación ----
    st.subheader("4. Interpretación")
    nivel, texto = interpretar(idx_c["Cpk"])
    getattr(st, nivel)(texto)
    if lei is not None and les is not None:
        medio = (lei + les) / 2
        if idx_c["Cp"] > 0 and (idx_c["Cp"] - idx_c["Cpk"]) / idx_c["Cp"] > 0.10:
            st.info(f"Cp ({idx_c['Cp']:.2f}) es bastante mayor que Cpk ({idx_c['Cpk']:.2f}): el proceso está "
                    f"descentrado {mu - medio:+.4f} respecto al centro de las especificaciones ({medio:.4f}). "
                    "Recentrarlo mejoraría la capacidad sin reducir la variabilidad.")
    if idx_p["Ppk"] < idx_c["Cpk"] - 0.15:
        st.info("Ppk es notablemente menor que Cpk: hay variación entre subgrupos (cambios de "
                "promedio en el tiempo) que σ dentro no refleja.")

    # Conclusión técnica integrada del módulo
    st.subheader("5. Conclusión técnica de capacidad")
    supuestos_previos = st.session_state.get("supuestos")
    estable = supuestos_previos is not None and supuestos_previos.get("senales", 0) == 0
    if idx_c["Cpk"] < 1:
        dictamen = "La capacidad estimada es insuficiente frente a las especificaciones: el índice Cpk es inferior a 1,00."
        accion = "Priorizar la reducción de variabilidad y/o el centrado del proceso, e investigar las fuentes de variación antes de establecer compromisos de conformidad."
    elif idx_c["Cpk"] < 1.33:
        dictamen = "El proceso presenta una capacidad marginal; aunque el desempeño puede acercarse a los requisitos, el margen disponible es limitado."
        accion = "Mantener seguimiento estrecho, evaluar el centrado y reducir la dispersión para alcanzar un margen de capacidad más robusto."
    elif idx_c["Cpk"] < 1.67:
        dictamen = "El proceso alcanza un nivel de capacidad generalmente aceptable bajo el criterio de referencia Cpk ≥ 1,33."
        accion = "Conservar los controles actuales y verificar que el desempeño se sostenga en el tiempo y en diferentes condiciones operativas."
    else:
        dictamen = "El proceso presenta una capacidad elevada respecto a los límites de especificación considerados."
        accion = "Mantener el sistema de control y confirmar periódicamente que las condiciones del proceso y las especificaciones continúan vigentes."
    if supuestos_previos is None:
        estabilidad_txt = "No se ha ejecutado la validación de supuestos en esta sesión; la capacidad debe interpretarse de forma provisional hasta evaluar estabilidad e independencia."
    elif estable:
        estabilidad_txt = "La evaluación preliminar de estabilidad no registró señales."
    else:
        estabilidad_txt = "Se registraron señales de posible inestabilidad; por tanto, la interpretación de capacidad debe considerarse provisional hasta investigar las causas especiales."
    centrado_txt = (f" La diferencia entre Cp y Cpk ({idx_c['Cp']-idx_c['Cpk']:.3f}) sugiere efecto de descentrado." if idx_c.get("Cp") is not None and idx_c["Cp"]-idx_c["Cpk"] > 0.10 else "")
    st.info(f"{dictamen} {estabilidad_txt}{centrado_txt} La fracción no conforme esperada con la dispersión global es {(bg+ag)*100:.4f} % ({(bg+ag)*1e6:.1f} PPM). {accion}")

    st.session_state["capacidad"] = {
        "lei": lei, "les": les, "mu": mu, "sigma_dentro": sw, "sigma_global": so,
        "Cp": idx_c.get("Cp"), "Cpk": idx_c["Cpk"], "Pp": idx_p.get("Pp"), "Ppk": idx_p["Ppk"],
        "ppm_global": (bg + ag) * 1e6, "nivel": nivel, "texto": texto,
        "conclusion_tecnica": f"{dictamen} {estabilidad_txt}{centrado_txt} La fracción no conforme esperada con la dispersión global es {(bg+ag)*100:.4f} % ({(bg+ag)*1e6:.1f} PPM). {accion}",
        "transformada": transf, "lambda": lam, "tabla_indices": tabla_indices,
        "tabla_fnc": tabla_fnc, "x": x.to_numpy(float), "n_obs": nobs}