"""Módulo 4. Validación de supuestos: normalidad, homocedasticidad e independencia."""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots
from scipy import stats
from statsmodels.nonparametric.smoothers_lowess import lowess
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.stats.stattools import durbin_watson
from statsmodels.tsa.stattools import acf, pacf

from modules import m2_variables as m2
from utils.datos import datos_estables, tipo_grafico_por_defecto
from utils.graficos import AZUL, ROJO, VERDE, NARANJA, GRIS, estilo


# ---------------------------------------------------------------- normalidad
def anderson_darling(x, alfa):
    """Anderson-Darling. Devuelve (estadístico, p-valor o None, ¿rechaza?)."""
    r = stats.anderson(x, dist="norm")
    p = getattr(r, "pvalue", None)
    if p is not None:
        return float(r.statistic), float(p), float(p) < alfa
    niveles = list(r.significance_level)
    crit = r.critical_values[min(range(len(niveles)), key=lambda i: abs(niveles[i] - alfa * 100))]
    return float(r.statistic), None, float(r.statistic) > crit


def graf_hist(x):
    """Histograma con densidad estimada (KDE), normal ajustada y boxplot marginal."""
    sd = x.std(ddof=1)
    xs = np.linspace(x.min() - 0.5 * sd, x.max() + 0.5 * sd, 300)
    nb = int(min(40, max(8, len(np.histogram_bin_edges(x, "fd")) - 1)))
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, row_heights=[0.16, 0.84],
                        vertical_spacing=0.02)
    fig.add_trace(go.Box(x=x, orientation="h", marker_color=AZUL, boxpoints="outliers",
                         showlegend=False, name=""), row=1, col=1)
    fig.add_trace(go.Histogram(x=x, histnorm="probability density", nbinsx=nb, name="Datos",
                               marker=dict(color="rgba(91,75,219,.35)", line=dict(color="white", width=1))),
                  row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=stats.gaussian_kde(x)(xs), mode="lines", name="Densidad (KDE)",
                             line=dict(color=AZUL, width=2.5)), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs, y=stats.norm.pdf(xs, x.mean(), sd), mode="lines",
                             name="Normal ajustada", line=dict(color=ROJO, dash="dash", width=2)),
                  row=2, col=1)
    fig.update_yaxes(showticklabels=False, row=1, col=1)
    return estilo(fig, "Distribución de los datos", 400)


def graf_qq(x, alfa):
    """Gráfico Q-Q normal con banda de confianza; resalta los puntos fuera de la banda."""
    n = len(x)
    xs = np.sort(x)
    p = (np.arange(1, n + 1) - 0.5) / n
    q = stats.norm.ppf(p)
    mu, sd = x.mean(), x.std(ddof=1)
    ajuste = mu + sd * q
    z = stats.norm.ppf(1 - alfa / 2)
    se = sd / stats.norm.pdf(q) * np.sqrt(p * (1 - p) / n)
    lo, hi = ajuste - z * se, ajuste + z * se
    fuera = (xs < lo) | (xs > hi)
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=q, y=hi, mode="lines", line=dict(width=0), hoverinfo="skip",
                             showlegend=False))
    fig.add_trace(go.Scatter(x=q, y=lo, mode="lines", line=dict(width=0), fill="tonexty",
                             fillcolor="rgba(91,75,219,.12)", name=f"Banda {100 * (1 - alfa):g} %",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=q, y=ajuste, mode="lines", name="Normal teórica",
                             line=dict(color=ROJO, dash="dash")))
    fig.add_trace(go.Scatter(x=q, y=xs, mode="markers", name="Datos",
                             marker=dict(color=[ROJO if f else AZUL for f in fuera], size=7,
                                         line=dict(color="white", width=.5))))
    fig.update_xaxes(title_text="Cuantiles teóricos (z)")
    fig.update_yaxes(title_text="Cuantiles muestrales")
    return estilo(fig, "Gráfico Q-Q normal", 400)


# ---------------------------------------------------------------- homocedasticidad
def grupos_homocedasticidad(df_e, col):
    """Grupos para comparar varianzas: subgrupos, o bloques consecutivos si n = 1."""
    g = df_e.groupby("_subgrupo", sort=False)[col]
    if g.count().max() >= 2:
        grupos = [(str(k), v[col].to_numpy(float)) for k, v in df_e.groupby("_subgrupo", sort=False)]
        return [(a, b) for a, b in grupos if len(b) >= 2], "subgrupos"
    x = df_e[col].to_numpy(float)
    k = max(5, len(x) // 6)
    grupos = [(f"Bloque {i + 1}", x[i * k:(i + 1) * k]) for i in range((len(x) + k - 1) // k)]
    return [(a, b) for a, b in grupos if len(b) >= 2], f"bloques de {k} observaciones consecutivas"


def graf_box(grupos):
    """Boxplot por grupo con los puntos individuales."""
    paso = int(np.ceil(len(grupos) / 60))
    fig = go.Figure()
    for lab, a in grupos[::paso]:
        fig.add_trace(go.Box(y=a, name=lab, boxpoints="all", jitter=.4, pointpos=0, showlegend=False,
                             marker=dict(color=AZUL, size=4), line=dict(width=1.2, color=AZUL),
                             fillcolor="rgba(91,75,219,.15)"))
    fig.add_hline(y=float(np.median(np.concatenate([a for _, a in grupos]))), line_dash="dot",
                  line_color=VERDE, annotation_text="Mediana global")
    fig.update_xaxes(title_text="Grupo")
    return estilo(fig, "Dispersión por grupo", 380)


def graf_std(grupos, alfa):
    """Desviación estándar por grupo con límites de probabilidad (chi-cuadrado)."""
    etq = [g for g, _ in grupos]
    s = np.array([a.std(ddof=1) for _, a in grupos])
    nu = np.array([len(a) - 1 for _, a in grupos], float)
    sp = float(np.sqrt(np.sum(nu * s ** 2) / np.sum(nu)))
    lo = sp * np.sqrt(stats.chi2.ppf(alfa / 2, nu) / nu)
    hi = sp * np.sqrt(stats.chi2.ppf(1 - alfa / 2, nu) / nu)
    fuera = (s < lo) | (s > hi)
    fig = go.Figure(go.Bar(x=etq, y=s, name="s del grupo",
                           marker_color=[ROJO if f else AZUL for f in fuera]))
    for y, nom, color, dash in ((hi, "Límite superior", ROJO, "dash"), (lo, "Límite inferior", ROJO, "dash")):
        fig.add_trace(go.Scatter(x=etq, y=y, mode="lines", name=nom,
                                 line=dict(color=color, dash=dash, shape="hv")))
    fig.add_trace(go.Scatter(x=etq, y=np.full(len(etq), sp), mode="lines", name="s combinada",
                             line=dict(color=VERDE)))
    fig.update_xaxes(title_text="Grupo")
    fig.update_yaxes(title_text="Desviación estándar")
    return estilo(fig, f"s por grupo (límites con probabilidad {100 * (1 - alfa):g} %)", 380)


def graf_residuos(grupos):
    """Residuos vs. media del grupo, con envolvente suavizada de la dispersión."""
    f = np.concatenate([np.full(len(a), a.mean()) for _, a in grupos])
    r = np.concatenate([a - a.mean() for _, a in grupos])
    fig = go.Figure(go.Scatter(x=f, y=r, mode="markers", name="Residuos",
                               marker=dict(color=AZUL, size=7, opacity=.7)))
    try:
        lw = lowess(np.abs(r), f, frac=0.8, return_sorted=True)
        env = 1.2533 * lw[:, 1]  # media de |r| × 1,2533 ≈ σ
        for signo, nom in ((1, "Dispersión suavizada (±)"), (-1, None)):
            fig.add_trace(go.Scatter(x=lw[:, 0], y=signo * env, mode="lines", name=nom or "",
                                     showlegend=nom is not None,
                                     line=dict(color=NARANJA, width=3)))
    except Exception:
        pass
    fig.add_hline(y=0, line_color=GRIS)
    fig.update_xaxes(title_text="Media del grupo")
    fig.update_yaxes(title_text="Residuo (dato − media del grupo)")
    return estilo(fig, "Residuos vs. nivel del proceso", 380)


# ---------------------------------------------------------------- independencia
def prueba_rachas(x):
    """Prueba de rachas (arriba/abajo de la mediana). Devuelve (rachas, z, p) o None."""
    x = np.asarray(x, float)
    s = x[x != np.median(x)] > np.median(x)
    n1, n2 = int(s.sum()), int(len(s) - s.sum())
    if n1 == 0 or n2 == 0 or len(s) < 10:
        return None
    rachas = 1 + int(np.sum(s[1:] != s[:-1]))
    mu = 2 * n1 * n2 / (n1 + n2) + 1
    var = (mu - 1) * (mu - 2) / (n1 + n2 - 1)
    if var <= 0:
        return None
    z = (rachas - mu) / np.sqrt(var)
    return rachas, float(z), float(2 * (1 - stats.norm.cdf(abs(z))))


def graf_corr(vals, ic, titulo):
    """Barras de ACF/PACF con banda de confianza; rojo = fuera de la banda."""
    lags = np.arange(1, len(vals) + 1)
    fig = go.Figure()
    fig.add_hrect(y0=-ic, y1=ic, fillcolor="rgba(91,75,219,.10)", line_width=0)
    fig.add_trace(go.Bar(x=lags, y=vals, width=0.25, showlegend=False,
                         marker_color=[ROJO if abs(v) > ic else AZUL for v in vals]))
    fig.add_trace(go.Scatter(x=lags, y=vals, mode="markers", showlegend=False,
                             marker=dict(color=[ROJO if abs(v) > ic else AZUL for v in vals], size=8)))
    fig.update_xaxes(title_text="Rezago", dtick=2)
    fig.update_yaxes(range=[-1, 1])
    return estilo(fig, titulo, 330)


def graf_corridas(s):
    """Gráfico de corridas con la mediana."""
    med = float(np.median(s))
    idx = np.arange(1, len(s) + 1)
    fig = go.Figure(go.Scatter(x=idx, y=s, mode="lines+markers", line=dict(color=GRIS, width=1),
                               marker=dict(color=[NARANJA if v > med else AZUL for v in s], size=8),
                               showlegend=False))
    fig.add_hline(y=med, line_dash="dash", line_color=VERDE, annotation_text="Mediana")
    fig.update_xaxes(title_text="Orden")
    return estilo(fig, "Gráfico de corridas", 330)


def graf_rezago(s):
    """Gráfico de dispersión de x(t) contra x(t−1)."""
    a, b = s[:-1], s[1:]
    r = float(np.corrcoef(a, b)[0, 1])
    m, c0 = np.polyfit(a, b, 1)
    xs = np.array([a.min(), a.max()])
    fig = go.Figure(go.Scatter(x=a, y=b, mode="markers", showlegend=False,
                               marker=dict(color=AZUL, size=8, opacity=.75)))
    fig.add_trace(go.Scatter(x=xs, y=m * xs + c0, mode="lines", line=dict(color=ROJO, dash="dash"),
                             showlegend=False))
    fig.update_xaxes(title_text="x(t−1)")
    fig.update_yaxes(title_text="x(t)")
    return estilo(fig, f"Gráfico de rezago 1 (r = {r:.3f})", 330)


def graf_tendencia(y, lim):
    """Evolución de la serie principal con tendencia suavizada (LOWESS)."""
    x = np.arange(1, len(y) + 1)
    lcl, cl, ucl = lim
    fig = go.Figure(go.Scatter(x=x, y=y, mode="lines+markers", name="Serie",
                               line=dict(color=AZUL, width=1.5), marker=dict(size=7)))
    try:
        lw = lowess(y, x, frac=0.5)
        fig.add_trace(go.Scatter(x=lw[:, 0], y=lw[:, 1], mode="lines", name="Tendencia (LOWESS)",
                                 line=dict(color=NARANJA, width=3)))
    except Exception:
        pass
    for v, nom, color, dash in ((ucl, "LCS", ROJO, "dash"), (cl, "LC", VERDE, "solid"), (lcl, "LCI", ROJO, "dash")):
        fig.add_hline(y=v, line_color=color, line_dash=dash, annotation_text=nom)
    fig.update_xaxes(title_text="Subgrupo / observación")
    return estilo(fig, "Evolución del proceso y tendencia suavizada", 360, derecha=50)


# ---------------------------------------------------------------- pantalla
def mostrar():
    st.header("Módulo 4. Validación de supuestos")
    df = st.session_state.get("df")
    col = st.session_state.get("col_valor")
    if df is None:
        st.warning("Primero cargue datos en **Gestión de datos** (Módulo 1).")
        return

    lim_m2 = st.session_state.get("limites")
    hay_excl = bool(lim_m2 and lim_m2.get("excluidos"))
    with st.container(border=True):
        c = st.columns([1.4, 1.3, 1.4, 1])
        usar = c[0].checkbox("Excluir puntos retirados en Fase I", value=True, disabled=not hay_excl)
        serie_op = c[1].selectbox("Serie para independencia",
                                  ["Observaciones individuales", "Medias de subgrupo"])
        alfa = c[2].select_slider("Significancia α", options=[0.01, 0.025, 0.05, 0.10], value=0.05)
        max_lag = int(c[3].number_input("Rezagos máx.", 5, 40, 20, 1))

    df_e = datos_estables(df, usar and hay_excl)
    x = df_e[col].dropna().to_numpy(float)
    if len(x) < 12:
        st.error("Se necesitan al menos 12 observaciones para validar supuestos.")
        return
    st.caption(f"Variable: **{col}** | Observaciones evaluadas: {len(x)} | α = {alfa}")
    resumen = {"n": len(x), "shapiro_p": float("nan"), "ad_p": float("nan"), "ad_rechaza": False, "ljung_p": float("nan"),
               "dw": float("nan"), "rachas_p": None, "serie": serie_op, "senales": 0, "alfa": alfa}

    t1, t2, t3, t4, t5 = st.tabs(["Normalidad", "Homocedasticidad", "Independencia",
                                  "Estabilidad", "Causas especiales"])

    # ---------------- Normalidad ----------------
    with t1:
        muestra = x if len(x) <= 5000 else np.random.default_rng(1).choice(x, 5000, replace=False)
        w, p_sw = stats.shapiro(muestra)
        ad, p_ad, rech_ad = anderson_darling(x, alfa)
        st.dataframe(pd.DataFrame([
            {"Prueba": "Shapiro-Wilk", "Estadístico": round(float(w), 4), "p-valor": round(float(p_sw), 4),
             f"Decisión (α = {alfa})": "Se rechaza normalidad" if p_sw < alfa else "No se rechaza normalidad"},
            {"Prueba": "Anderson-Darling", "Estadístico": round(ad, 4),
             "p-valor": round(p_ad, 4) if p_ad is not None else None,
             f"Decisión (α = {alfa})": "Se rechaza normalidad" if rech_ad else "No se rechaza normalidad"},
        ]), hide_index=True)
        k1, k2 = st.columns(2)
        k1.plotly_chart(graf_hist(x), width="stretch", key="g_hist")
        k2.plotly_chart(graf_qq(x, alfa), width="stretch", key="g_qq")
        st.caption("Q-Q: si los puntos siguen la línea y quedan dentro de la banda, los datos son "
                   "compatibles con la normal. Las colas que se separan indican datos atípicos o asimetría.")
        if p_sw < alfa or rech_ad:
            st.warning("Hay evidencia contra la normalidad. Revise atípicos o mezcla de condiciones "
                       "(Módulo 2) y considere Box-Cox en el Módulo 5.")
        else:
            st.success("No hay evidencia contra la normalidad.")
        st.markdown("**Conclusión del supuesto**")
        st.write((f"Con α = {alfa:.3f}, al menos una prueba aporta evidencia estadística contra la normalidad. "
                  "La distribución puede afectar la validez de métodos que requieren normalidad; conviene revisar asimetría, colas y observaciones atípicas." )
                 if (p_sw < alfa or rech_ad) else
                 f"Con α = {alfa:.3f}, las pruebas aplicadas no aportan evidencia suficiente para rechazar la normalidad. Esto no demuestra normalidad absoluta; debe complementarse con la inspección del histograma y el gráfico Q-Q.")
        resumen.update({"shapiro_p": float(p_sw), "ad_p": float(p_ad) if p_ad is not None else float("nan"), "ad_rechaza": bool(rech_ad)})

    # ---------------- Homocedasticidad ----------------
    with t2:
        grupos, base = grupos_homocedasticidad(df_e, col)
        if len(grupos) < 2:
            st.info("Se necesitan al menos 2 grupos con 2 o más observaciones.")
        else:
            arrs = [a for _, a in grupos]
            st.caption(f"Se comparan {len(grupos)} grupos ({base}).")
            pruebas = [("Levene (mediana, robusta)", lambda: stats.levene(*arrs, center="median")),
                       ("Bartlett (supone normalidad)", lambda: stats.bartlett(*arrs)),
                       ("Fligner-Killeen (no paramétrica)", lambda: stats.fligner(*arrs))]
            filas, pv = [], {}
            for nombre, f in pruebas:
                try:
                    r = f()
                    pv[nombre] = float(r.pvalue)
                    filas.append({"Prueba": nombre, "Estadístico": round(float(r.statistic), 4),
                                  "p-valor": round(float(r.pvalue), 4),
                                  f"Decisión (α = {alfa})": "Varianzas distintas" if r.pvalue < alfa
                                  else "Varianzas homogéneas (no se rechaza)"})
                except Exception:
                    filas.append({"Prueba": nombre, "Estadístico": None, "p-valor": None,
                                  f"Decisión (α = {alfa})": "No calculable"})
            st.dataframe(pd.DataFrame(filas), hide_index=True)
            k1, k2 = st.columns(2)
            k1.plotly_chart(graf_box(grupos), width="stretch", key="g_box")
            k2.plotly_chart(graf_std(grupos, alfa), width="stretch", key="g_std")
            k3, k4 = st.columns([1.4, 1])
            k3.plotly_chart(graf_residuos(grupos), width="stretch", key="g_res")
            with k4:
                if len(grupos) >= 4:
                    rho, p_r = stats.spearmanr([a.mean() for a in arrs], [a.std(ddof=1) for a in arrs])
                    st.write(f"**Dispersión vs. nivel** (Spearman entre media y s de cada grupo): "
                             f"ρ = {rho:.3f}, p = {p_r:.4f} → "
                             f"{'la variabilidad depende del nivel' if p_r < alfa else 'sin relación significativa'}.")
                st.caption("Si los residuos forman un embudo (más abiertos a un lado) o la envolvente "
                           "naranja sube o baja, la varianza no es constante.")
            if any(p < alfa for p in pv.values()):
                st.warning("Hay evidencia de varianzas desiguales: los límites de los gráficos pueden ser "
                           "poco representativos. Busque cambios de máquina, turno, lote u operario.")
            else:
                st.success("No hay evidencia contra la homocedasticidad.")
            resumen["levene_p"] = pv.get("Levene (mediana, robusta)")
            resumen["bartlett_p"] = pv.get("Bartlett (supone normalidad)")
            st.markdown("**Conclusión del supuesto**")
            hom_rechazo = any(pv.get(k, 1) < alfa for k in pv)
            st.write(("Se identifica evidencia de diferencias entre varianzas de los grupos. La heterogeneidad puede comprometer la interpretación de límites y comparaciones; se recomienda segmentar por turno, máquina, lote u otra condición operativa relevante." if hom_rechazo else "Las pruebas no detectan diferencias estadísticamente significativas entre las varianzas evaluadas. Los resultados son compatibles con homocedasticidad para los grupos y datos analizados."))

    # ---------------- Independencia ----------------
    with t3:
        if serie_op.startswith("Medias"):
            s = df_e.groupby("_subgrupo", sort=False)[col].mean().to_numpy(float)
        else:
            s = x
        if len(s) < 12:
            st.error("La serie elegida tiene menos de 12 puntos.")
        else:
            nlags = int(min(max_lag, len(s) // 2 - 1))
            ic = float(stats.norm.ppf(1 - alfa / 2) / np.sqrt(len(s)))
            k1, k2 = st.columns(2)
            k1.plotly_chart(graf_corr(acf(s, nlags=nlags, fft=True)[1:], ic,
                                      "Autocorrelación (ACF)"), width="stretch", key="g_acf")
            k2.plotly_chart(graf_corr(pacf(s, nlags=nlags, method="ywm")[1:], ic,
                                      "Autocorrelación parcial (PACF)"), width="stretch", key="g_pacf")
            k3, k4 = st.columns(2)
            k3.plotly_chart(graf_corridas(s), width="stretch", key="g_runs")
            k4.plotly_chart(graf_rezago(s), width="stretch", key="g_lag")
            st.caption("Barras rojas = fuera de la banda de confianza. En un proceso independiente casi "
                       "todas deben quedar dentro (alguna aislada puede salir por azar).")

            lags = max(1, int(min(10, len(s) // 5)))
            lb = acorr_ljungbox(s, lags=[lags], return_df=True)
            p_lb = float(lb["lb_pvalue"].iloc[0])
            dw = float(durbin_watson(s - s.mean()))
            rr = prueba_rachas(s)
            filas = [
                {"Prueba": f"Ljung-Box (rezagos = {lags})", "Estadístico": round(float(lb["lb_stat"].iloc[0]), 4),
                 "p-valor": round(p_lb, 4),
                 f"Decisión (α = {alfa})": "Hay autocorrelación" if p_lb < alfa else "Sin evidencia de autocorrelación"},
                {"Prueba": "Durbin-Watson (≈2 = independiente)", "Estadístico": round(dw, 4), "p-valor": None,
                 f"Decisión (α = {alfa})": "Posible autocorrelación positiva" if dw < 1.5
                 else ("Posible autocorrelación negativa" if dw > 2.5 else "Sin evidencia de autocorrelación")},
            ]
            p_rachas = None
            if rr:
                p_rachas = rr[2]
                filas.append({"Prueba": f"Rachas (mediana): {rr[0]} rachas", "Estadístico": round(rr[1], 4),
                              "p-valor": round(rr[2], 4),
                              f"Decisión (α = {alfa})": "Patrón no aleatorio" if rr[2] < alfa else "Aleatorio"})
            st.dataframe(pd.DataFrame(filas), hide_index=True)
            resumen.update({"ljung_p": p_lb, "dw": dw, "rachas_p": p_rachas})
            evidencia_ind = p_lb < alfa or (p_rachas is not None and p_rachas < alfa) or dw < 1.5 or dw > 2.5
            st.markdown("**Conclusión del supuesto**")
            st.write(("Se observan indicios de dependencia temporal o patrones no aleatorios. En estas condiciones, las observaciones consecutivas podrían no ser independientes y la estimación de límites o pruebas posteriores puede verse afectada. Investigue ciclos, turnos, lotes y secuencias de operación." if evidencia_ind else "Las pruebas aplicadas no muestran evidencia estadística suficiente de dependencia temporal. La serie resulta compatible con independencia en el alcance evaluado; mantenga la vigilancia de patrones en los gráficos ACF, PACF y de rachas."))

    # ---------------- Estabilidad ----------------
    with t4:
        tipo = tipo_grafico_por_defecto(df, col)
        principal, disp, n_sg = m2.construir_series(df, col, tipo)
        total = 0
        if len(principal) >= 8:
            lim = m2.calcular_limites(tipo, principal, disp, n_sg)
            sen_p = m2.evaluar_senales(principal, lim["principal"], True)
            sen_d = m2.evaluar_senales(disp, lim["disp"], False)
            filas = [{"Regla": m2.DESCRIPCION[c_], "Gráfico": m2.NOMBRES[tipo][0],
                      "Puntos": len(v), "Cuáles": ", ".join(map(str, v)) or "-"} for c_, v in sen_p.items()]
            filas.append({"Regla": m2.DESCRIPCION["R1"], "Gráfico": m2.NOMBRES[tipo][1],
                          "Puntos": len(sen_d["R1"]), "Cuáles": ", ".join(map(str, sen_d["R1"])) or "-"})
            total = sum(f["Puntos"] for f in filas)
            st.caption(f"Gráfico usado: {tipo}, con límites calculados con todos los datos (sin depurar).")
            st.plotly_chart(graf_tendencia(principal.values, lim["principal"]), width="stretch",
                            key="g_tend")
            st.dataframe(pd.DataFrame(filas), hide_index=True)
            rho, p_sp = stats.spearmanr(np.arange(len(principal)), principal.values)
            st.write(f"Tendencia (Spearman entre orden y valor): ρ = {rho:.3f}, p = {p_sp:.4f} → "
                     f"{'**hay tendencia**' if p_sp < alfa else 'sin tendencia significativa'}.")
            resumen["spearman_p"] = float(p_sp)
            if total == 0:
                st.success("Sin señales: el proceso parece estable con los datos disponibles.")
            else:
                st.warning(f"{total} señal(es) de falta de control. Investigue y registre las causas "
                           "especiales en la pestaña siguiente; depure en el Módulo 2.")
        else:
            st.info("Muy pocos subgrupos para evaluar estabilidad.")
        resumen["senales"] = total
        st.markdown("**Conclusión del supuesto**")
        st.write((f"Se detectaron {total} señal(es) asociadas con reglas de control. El proceso no debe considerarse estadísticamente estable hasta investigar y documentar las causas especiales y recalcular los límites cuando corresponda." if total else "No se detectaron señales mediante las reglas configuradas. La evidencia disponible es compatible con estabilidad estadística preliminar, sin que ello garantice el comportamiento futuro."))

    # ---------------- Causas especiales ----------------
    with t5:
        st.caption("Escriba cada causa identificada en la visita a la empresa. Se exporta al reporte.")
        base = pd.DataFrame({"Subgrupo / punto": [""], "Causa especial identificada": [""],
                             "Acción correctiva": [""]})
        st.session_state["causas_tabla"] = st.data_editor(base, num_rows="dynamic", hide_index=True,
                                                          key="tabla_causas")
        st.caption("Documente la evidencia operativa que permita explicar cada señal. No atribuya una causa sin verificación.")

    st.session_state["supuestos"] = resumen